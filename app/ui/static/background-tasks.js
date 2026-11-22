// Session-scoped task state. The popup supplies its authenticated transport.
class BackgroundTasks {
  constructor({ list, changed, active, setTimer = (callback, delay) => setTimeout(callback, delay),
    clearTimer = (timer) => clearTimeout(timer) }) {
    Object.assign(this, { list, changed, active, setTimer, clearTimer });
    this.scope = null;
    this.tasks = new Map();
    this.timer = null;
    this.loading = null;
    this.remaining = 0;
    this.selection = null;
  }

  configure(session, epoch) {
    if ((!epoch && !this.scope) || (this.scope && session === this.scope.session && epoch === this.scope.epoch)) return;
    this.stop();
    const sameSession = session === this.scope?.session;
    this.scope = epoch ? { session, epoch } : null;
    this.selection = null;
    this.loading = null;
    if (!sameSession || !epoch) this.tasks.clear();
    else for (const [id, task] of this.tasks) this.tasks.set(id, { ...task, status: "expired" });
    this.remaining = 24;
    this.changed();
    if (this.scope) void this.refresh();
  }

  stop() {
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = null;
  }

  remember(task) {
    if (!this.scope || !task?.task_id || task.runtime_epoch !== this.scope.epoch) return false;
    const prior = this.tasks.get(task.task_id);
    const status = task.status === "completed" ? "ready" : task.status;
    const version = Number(task.status_version || 1);
    if (prior && (prior.status === "expired" || version < prior.status_version)) return false;
    if (prior?.status === status && prior.status_version === version && (!task.label || prior.label === task.label)) {
      return false;
    }
    if (!prior && ["queued", "loading", "running", "cancelling"].includes(status)) this.remaining = 24;
    this.tasks.set(task.task_id, { ...prior, ...task, status, status_version: version });
    while (this.tasks.size > 32) this.tasks.delete(this.tasks.keys().next().value);
    if (this.selection && this.tasks.get(this.selection)?.status !== "ready") this.selection = null;
    return true;
  }

  receipt(task) {
    if (this.remember(task)) this.changed();
    this.schedule();
  }

  select(id) {
    this.selection = this.tasks.get(id)?.status === "ready" ? id : null;
    this.changed();
  }

  selected() {
    const task = this.tasks.get(this.selection);
    return task && { task_id: task.task_id, runtime_epoch: task.runtime_epoch };
  }

  schedule() {
    if (this.timer !== null || this.loading || !this.scope || !this.active() || !this.remaining
        || ![...this.tasks.values()].some((row) => ["queued", "loading", "running", "cancelling"].includes(row.status))) return;
    this.timer = this.setTimer(() => {
      this.timer = null;
      if (!this.active()) return;
      this.remaining--;
      void this.refresh();
    }, 2500);
  }

  async refresh() {
    if (!this.scope || !this.active() || this.loading) return;
    this.stop();
    const scope = this.scope;
    this.loading = scope;
    const before = new Map(this.tasks);
    try {
      const payload = await this.list();
      if (this.scope !== scope || !payload) return;
      if (payload.runtime_epoch !== scope.epoch) {
        this.configure(scope.session, payload.runtime_epoch);
        return;
      }
      const received = new Set();
      let changed = false;
      for (const task of payload.tasks || []) {
        received.add(task.task_id);
        changed = this.remember(task) || changed;
      }
      for (const [id, prior] of before) {
        if (this.tasks.get(id) === prior && prior.status !== "expired" && !received.has(id)) {
          this.tasks.set(id, { ...prior, status: "expired" });
          if (this.selection === id) this.selection = null;
          changed = true;
        }
      }
      if (changed) this.changed();
    } catch {
      // The bounded pending poll can retry transient network failures.
    } finally {
      if (this.loading === scope) {
        this.loading = null;
        this.schedule();
      }
    }
  }
}

if (typeof module !== "undefined") module.exports = BackgroundTasks;
