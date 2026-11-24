"""Host pressure checks for an opt-in cold L2 load on macOS."""

from dataclasses import dataclass, field
import re
import subprocess
from threading import Event, Thread
from time import monotonic


_VM_RESERVE = 5 * 2**30
# Native K/V, recurrent, compute and transient allocations are separate from the mapped GGUF.
_COLD_LOAD_HEADROOM = 2**30


class L2ResourceError(RuntimeError):
    pass


class L2LoadTimeout(L2ResourceError):
    def __init__(self, deadline):
        self.deadline_monotonic = deadline
        super().__init__('The L2 cold-start deadline expired.')


@dataclass(frozen=True, slots=True)
class MemoryReading:
    available: int
    pressure: int
    swap_mib: float
    pageout_mib: float
    available_vm_bytes: int | None = field(default=None, kw_only=True)


def mac_memory() -> MemoryReading:
    def query(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()

    try:
        available = re.search(r'free percentage:\s*(\d+)%', query('/usr/bin/memory_pressure', '-Q'))
        pressure = int(query('/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'))
        swap = re.search(r'used\s*=\s*([\d.]+)M', query('/usr/sbin/sysctl', '-n', 'vm.swapusage'))
        stats = query('/usr/bin/vm_stat')
        page_size = re.search(r'page size of (\d+) bytes', stats)
        pageouts = re.search(r'^Pageouts:\s+(\d+)\.', stats, re.M)
        pool = re.findall(r'^Pages (free|active|inactive|speculative):\s*(\d+)\.$', stats, re.M)
        if not (available and swap and page_size and pageouts and len(pool) == len(dict(pool)) == 4):
            raise ValueError('Incomplete memory readings.')
        page_bytes = int(page_size[1])
        return MemoryReading(int(available[1]), pressure, float(swap[1]), page_bytes * int(pageouts[1]) / 2**20,
                             available_vm_bytes=page_bytes * sum(int(count) for _, count in pool))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise L2ResourceError('Required macOS memory counters are unavailable.') from error


def check_memory(reading, baseline, *, floor, swap_limit):
    if reading.pressure not in (0, 1) or reading.available < floor:
        raise L2ResourceError(f'Memory admission requires normal OS pressure and at least {floor}% available.')
    if reading.swap_mib - baseline.swap_mib > swap_limit:
        raise L2ResourceError(f'Host swap increased by more than {swap_limit} MiB.')
    if reading.pageout_mib - baseline.pageout_mib > 128:
        raise L2ResourceError('Host pageouts increased by more than 128 MiB.')


def check_model_holders(path):
    try:
        result = subprocess.run(['/usr/sbin/lsof', '-t', str(path.resolve())], capture_output=True, timeout=2)
    except (OSError, subprocess.SubprocessError) as error:
        raise L2ResourceError('Could not check the Qwen GGUF for existing holders.') from error
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise L2ResourceError('The Qwen GGUF is in use or exclusive cold-load access is uncertain.')


def guarded_l2_load(factory, cancellation, on_abort, *, model_path=None, read=mac_memory, clock=monotonic, seconds=20.0,
                    quiet_seconds=1.5, interval=0.4, admit=None):
    """The native constructor observes the stop flag at its supported checkpoints."""
    finished, resource_stop = Event(), Event()
    deadline = clock() + seconds
    error = None
    entry_bytes = None

    def check_attempt():
        cancellation.raise_if_cancelled()
        if error is not None:
            raise error
        if clock() >= deadline:
            raise L2LoadTimeout(deadline)

    def check_quiet(first, reading):
        check_memory(reading, first, floor=31 if admit is None else 6, swap_limit=64)
        if admit is not None:
            admit(first, reading)
        if entry_bytes is not None:
            check_pool(reading, entry_bytes)

    def check_pool(reading, required):
        available = reading.available_vm_bytes
        if type(available) is not int or available < required:
            raise L2ResourceError(f'Qwen L2 requires {required} bytes in the VM pool. Observed {available} bytes.')

    def check_loading(reading):
        check_memory(reading, baseline, floor=6, swap_limit=256)
        if entry_bytes is not None:
            check_pool(reading, _VM_RESERVE)

    check_attempt()
    if model_path is not None:
        check_model_holders(model_path)
        if admit is None:
            try:
                entry_bytes = model_path.stat().st_size + _COLD_LOAD_HEADROOM + _VM_RESERVE
            except OSError as error:
                raise L2ResourceError('Could not size the Qwen GGUF for memory admission.') from error
    first = read()
    check_quiet(first, first)
    until = clock() + quiet_seconds
    while clock() < until:
        check_attempt()
        finished.wait(min(0.05, max(0, until - clock())))
    baseline = read()
    check_quiet(first, baseline)
    check_attempt()

    def monitor():
        nonlocal error
        while not finished.wait(interval):
            try:
                check_attempt()
                check_loading(read())
            except Exception as failure:
                error = failure
                resource_stop.set()
                on_abort()
                return

    watcher = Thread(target=monitor, name='akane-l2-load-guard', daemon=True)
    watcher.start()
    model = None
    try:
        try:
            model = factory(lambda: resource_stop.is_set() or cancellation.cancelled or clock() >= deadline)
        except Exception:
            check_attempt()  # A native abort can be reported as a constructor error.
            raise
        check_attempt()
        check_loading(read())
    except BaseException:
        if model is not None:
            model.close()
        raise
    finally:
        finished.set()
        watcher.join()
    try:
        check_attempt()  # Include any monitor failure observed during the final sample.
    except BaseException:
        model.close()
        raise
    return model
