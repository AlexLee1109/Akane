"""Akane Next over the real loopback transport and existing popup bridge."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
import runpy
import shutil
import socket
import sys
from threading import Event, Thread
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.companion import next_adapter, server
from app.core import config
from app.ui import popup
from app.ai.automatic_contracts import AutomaticContextPolicy
from app.ai.background.native import AUTOMATIC_FINDING_SCHEMA
from app.ai.contracts import HistoryEntry
from app.ai.conversation.character import load_l1_prompt
from app.ai.conversation.decisions import INTENT_ONLY_FINAL_SCHEMA, INTENT_ONLY_SCHEMA
from app.ai.conversation.model import PreparedPrompt


class Model:
    context_size = 4096
    model_path = Path("fixture.gguf")

    def __init__(self):
        self.prompts = []
        self.closed = False
        self.stopped = Event()

    def prepare(self, messages, *, max_tokens):
        text = "\n".join(message.content for message in messages)
        return PreparedPrompt(messages, text, (1,) * (len(text.encode()) // 3), max_tokens)

    def stream(self, prompt, *, cancellation):
        self.prompts.append(prompt)
        if prompt.json_schema in (INTENT_ONLY_SCHEMA, INTENT_ONLY_FINAL_SCHEMA):
            yield '{"mode":"reply","text":"First second"}'
        elif prompt.json_schema:
            yield '{"text":"User gave the fictional label; the assistant acknowledged it."}'
        elif prompt.messages[0].content.startswith("You are an informational task worker."):
            yield "Fixture finding from the supplied note."
        else:
            try:
                yield "First "
                if prompt.messages[-1].content == "Cancel this fixture turn.":
                    assert cancellation._signal.wait(5)
                    cancellation.raise_if_cancelled()
                if prompt.messages[-1].content == "Fail this fixture turn.":
                    raise ValueError("Fixture generator failed.")
                yield "second"
            finally:
                self.stopped.set()

    def close(self):
        self.closed = True


def http(url, path, data=None, method=None):
    return urlopen(Request(url + path, data=None if data is None else json.dumps(data).encode(),
                           headers={"Content-Type": "application/json",
                                    "Authorization": "Bearer isolated-test-credential"}, method=method), timeout=10)


def read(url, path, data=None, method=None):
    with http(url, path, data, method) as response:
        return json.load(response)


@contextmanager
def listener(companion):
    backend = server.BackgroundServer("127.0.0.1", 0, server.create_app(companion))
    url = "http://127.0.0.1:" + str(backend._socket.getsockname()[1])
    errors = []

    def run():
        try:
            backend.run()
        except Exception as error:
            errors.append(error)

    thread = Thread(target=run)
    backend._thread = thread
    thread.start()
    try:
        deadline = monotonic() + 5
        while thread.is_alive() and not backend._server.started and monotonic() < deadline:
            sleep(.01)
        assert backend._server.started, errors
        yield url
    finally:
        backend.shutdown()
    assert not errors


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path.resolve() / "next"
    monkeypatch.setenv("AKANE_NEXT_DATA_DIR", str(path))
    monkeypatch.delenv("NEXT_AUTO_DELEGATION", raising=False)
    monkeypatch.setattr(config, "_local_secrets", None)
    monkeypatch.setattr(next_adapter, "SETTINGS", replace(next_adapter.SETTINGS, next_auto_delegation=False))
    for module in (server, popup):
        monkeypatch.setattr(module, "SETTINGS", replace(module.SETTINGS, server_api_token="isolated-test-credential"))
    models = []

    def create(_path):
        model = Model()
        models.append(model)
        return model

    monkeypatch.setattr(next_adapter, "NativeModel", create)
    return path, models


def test_both_command_modes_load_the_current_application_entry_points(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "serve", lambda **_: calls.append("server"))
    monkeypatch.setattr(popup, "launch_popup", lambda: calls.append("popup"))
    for mode in ("server", "popup"):
        monkeypatch.setattr(sys, "argv", ["app", mode])
        runpy.run_module("app", run_name="__main__", alter_sys=True)
    assert calls == ["server", "popup"]
    from app.server import APP
    assert APP.state.companion.runtime_name == "akane_next"
    assert not {"app.core.inference", "app.companion.model", "app.companion.conversation"} & sys.modules.keys()


def test_popup_bridge_streams_real_http_and_reuses_server(runtime, monkeypatch):
    _, models = runtime
    companion = next_adapter.NextCompanion()
    with listener(companion) as url:
        assert read(url, "/health") == {"status": "ok", "runtime": "akane_next", "model_loaded": True, "model_loading": False}
        monkeypatch.setattr(popup, "SETTINGS", replace(popup.SETTINGS, popup_backend_url=url))
        client = popup.PopupApp()
        assert client.server is None
        received = []
        monkeypatch.setattr(client, "_emit_stream_event", received.append)
        client._run_message_stream("Fixture note.", "popup-test", True)
        assert [event["type"] for event in received] == ["start", "delta", "delta", "flush", "done"]
        assert received[-1]["reply"] == "First second"
        assert {event["request_id"] for event in received} == {"popup-test"}
        assert len({event["generation_id"] for event in received}) == 1
        assert "Fixture note." in models[0].prompts[0].text
        system = models[0].prompts[0].messages[0].content
        assert system.split("\n\nAttributed context data", 1)[0].rstrip() == load_l1_prompt().rstrip()
        assert client.send_command("/reset_chat")["messages"] == []
    assert len(models) == 1 and models[0].closed


@pytest.mark.parametrize("override, expected_schema", [(None, INTENT_ONLY_SCHEMA), ("true", INTENT_ONLY_SCHEMA), ("false", None)])
def test_popup_alone_starts_and_drains_its_own_akane_next_server(runtime, monkeypatch, override, expected_schema):
    _, models = runtime
    if override is None:
        monkeypatch.delenv("AKANE_NEXT_AUTO_DELEGATION", raising=False)
    else:
        monkeypatch.setenv("AKANE_NEXT_AUTO_DELEGATION", override)
    monkeypatch.setattr(next_adapter, "SETTINGS", config.load_settings())
    with socket.create_server(("127.0.0.1", 0)) as reserved:
        port = reserved.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(popup, "SETTINGS", replace(popup.SETTINGS, popup_backend_url=url))
    monkeypatch.setattr(popup, "popup_backend_is_local", lambda: True)
    client = popup.PopupApp()
    try:
        assert client.server is not None
        assert read(url, "/health")["runtime"] == "akane_next"
        message = ("Independent popup launch. Describe the fictional note I mentioned earlier, "
                   "remembering that I have yet to tell you its name or contents and only want to discuss it.")
        assert read(url, "/api/chat", {"message": message})["outcome"] == "completed"
        assert [prompt.json_schema for prompt in models[0].prompts] == [expected_schema]
    finally:
        client._cleanup_background_resources()
    assert len(models) == 1 and models[0].closed


def test_foreground_cancel_partial_failure_recovery_and_restart(runtime):
    _, models = runtime
    with listener(next_adapter.NextCompanion()) as url:
        with http(url, "/api/chat/stream", {"message": "Cancel this fixture turn.", "request_id": "a"}) as stream:
            start, first = (json.loads(stream.readline()) for _ in range(2))
            assert first["type"] == "delta"
            assert not read(url, "/api/chat/cancel", {"request_id": "stale"})["cancelled"]
            assert read(url, "/api/chat/cancel", {"generation_id": start["generation_id"]})["cancelled"]
            cancelled = json.loads(stream.readline())
            assert (cancelled["type"], cancelled["reply"]) == ("cancelled", "First ")
            assert cancelled["messages"][-1]["outcome"] == "cancelled"
        with http(url, "/api/chat/stream", {"message": "Fail this fixture turn."}) as stream:
            events = list(map(json.loads, stream))
        assert [e["type"] for e in events] == ["start", "delta", "error"]
        assert events[-1]["messages"][-1]["outcome"] == "failed"
        assert read(url, "/api/chat", {"message": "Recover."})["outcome"] == "completed"
        assert read(url, "/api/state?session_id=other")["messages"] == []
        assert read(url, "/api/session", {"session_id": "other"})["revision"] == 0
        assert not read(url, "/api/context?session_id=popup")["active"]
        assert read(url, "/api/context?session_id=other")["active"]
    with listener(next_adapter.NextCompanion()) as url:
        assert [row["outcome"] for row in read(url, "/api/state")["messages"][1::2]] == [
            "cancelled", "failed", "completed",
        ]
    assert len(models) == 2 and all(model.closed for model in models)


def test_disconnected_stream_cancels_the_actual_turn(runtime):
    _, models = runtime
    companion = next_adapter.NextCompanion()
    with listener(companion) as url:
        with http(url, "/api/chat/stream", {"message": "Cancel this fixture turn."}) as stream:
            assert json.loads(stream.readline())["type"] == "start"
            assert json.loads(stream.readline())["type"] == "delta"
        assert models[0].stopped.wait(3)
        with companion._lock:
            assert companion._lock.wait_for(lambda: companion._active is None, 3)
        assert read(url, "/api/state")["messages"][-1]["outcome"] == "cancelled"


@pytest.mark.parametrize("automatic", [False, True])
def test_memory_approval_skip_and_manual_job_attachment(runtime, monkeypatch, automatic):
    _, models = runtime
    monkeypatch.setattr(next_adapter, "SETTINGS", replace(next_adapter.SETTINGS, next_auto_delegation=automatic))
    companion = next_adapter.NextCompanion()
    with listener(companion) as url:
        original = read(url, "/api/chat", {"message": "The fictional label is Birch."})["messages"][0]
        proposal = {"source_turn_id": original["id"], "text": original["content"]}
        with pytest.raises(HTTPError) as denied:
            read(url, "/api/memory", proposal)
        assert denied.value.code == 403
        denied.value.close()
        read(url, "/api/memory", {**proposal, "confirmed": True})
        read(url, "/api/chat", {"message": "Consult my approved quote."})
        assert "User-approved" in models[0].prompts[-1].text
        read(url, "/api/chat", {"message": "Skip memory on this turn.", "skip_memory": True})
        assert "User-approved" not in models[0].prompts[-1].text
        assert len(read(url, "/api/memory")["facts"]) == 1
        job = read(url, "/api/background", {"task": "Report the supplied label.", "source_turn_id": original["id"]})
        job_id = next_adapter.UUID(job["job_id"])
        companion._scheduler.wait_result(next_adapter._PROFILE_ID, job_id, timeout=5)
        route = "/api/background/" + str(job_id)
        completed = read(url, route)
        assert completed["status"] == "COMPLETED" and completed["findings"][0]["sources"][0]["owner"] == "l2"
        with pytest.raises(HTTPError) as other:
            read(url, route + "/attach", {"session_id": "other"})
        assert other.value.code == 404
        other.value.close()
        read(url, route + "/attach", {})
        final = read(url, "/api/chat", {"message": "Use the worker finding."})
        assert "Fixture finding" in models[0].prompts[-1].text
        assert models[0].prompts[-1].json_schema == (INTENT_ONLY_FINAL_SCHEMA if automatic else None)
        assert read(url, route)["delivered_turn_id"] == final["messages"][-1]["id"]
        read(url, "/api/chat", {"message": "Next turn."})
        assert "Fixture finding" not in models[0].prompts[-1].text


def test_default_delegation_streams_once_and_blocks_replayed_and_private_jobs(runtime, monkeypatch, capsys):
    _, models = runtime
    user_text = ('Compare synthetic crates with the private marker R17. We need capacity for four boxes '
                 'and enough clearance to fit through the doorway. Only the blue crate fits.')
    finding = 'The blue crate fits because it holds enough.'

    class IntentModel(Model):
        worker_output = json.dumps({'text': finding})

        def stream(self, prompt, *, cancellation):
            self.prompts.append(prompt)
            if prompt.json_schema == INTENT_ONLY_SCHEMA:
                if prompt.messages[-1].content == user_text:
                    yield '{"mode":"delegate","context_refs":["u0"]}'
                else:
                    yield '{"mode":"reply","text":"Direct reply."}'
            elif prompt.json_schema == AUTOMATIC_FINDING_SCHEMA:
                source = json.loads(prompt.messages[-1].content)
                assert source['task'] == user_text == source['sources'][0]['text']
                yield self.worker_output
            elif prompt.json_schema is None:
                yield 'Direct reply.'
            else:
                assert prompt.json_schema == INTENT_ONLY_FINAL_SCHEMA and finding in prompt.text
                yield '{"mode":"reply","text":"Choose the blue crate.'
                yield '"}'

    model = IntentModel()
    models.append(model)
    monkeypatch.setattr(next_adapter, 'NativeModel', lambda _path: model)
    monkeypatch.delenv('AKANE_NEXT_AUTO_DELEGATION', raising=False)
    monkeypatch.setattr(next_adapter, 'SETTINGS', replace(config.load_settings(), timing_enabled=True))
    companion = next_adapter.NextCompanion()
    with listener(companion) as url:
        body = {'message': user_text, 'request_id': 'same-client-request'}
        with http(url, '/api/chat/stream', body) as response:
            events = [json.loads(row) for row in response]
        assert events[0]['type'] == 'start' and events[-1]['type'] == 'done'
        assert ''.join(row['content'] for row in events if row['type'] == 'delta') == 'Choose the blue crate.'
        assert [prompt.json_schema for prompt in model.prompts] == [INTENT_ONLY_SCHEMA, AUTOMATIC_FINDING_SCHEMA, INTENT_ONLY_FINAL_SCHEMA]
        with pytest.raises(HTTPError) as private:
            read(url, '/api/background/' + events[0]['generation_id'])
        assert private.value.code == 404
        private.value.close()
        with pytest.raises(HTTPError) as duplicate:
            read(url, '/api/chat', body)
        assert duplicate.value.code == 409
        duplicate.value.close()
        assert len(model.prompts) == 3
        assert [row['content'] for row in read(url, '/api/state')['messages']] == [user_text, 'Choose the blue crate.']
        assert read(url, '/api/chat', {'message': 'Thanks.', 'request_id': 'new-request'})['reply'] == 'Direct reply.'
        assert model.prompts[-1].json_schema is None
        assert finding not in model.prompts[-1].text
        assert read(url, '/api/chat', {**body, 'request_id': 'new-genuine-turn'})['revision'] == 3
        assert read(url, '/api/chat', {**body, 'session_id': 'other'})['revision'] == 1

        model.worker_output = '{"text":"unfinished'
        previous_calls = len(model.prompts)
        monkeypatch.setattr(popup, 'SETTINGS', replace(popup.SETTINGS, popup_backend_url=url))
        client = popup.PopupApp()
        assert client.server is None
        failed = []
        monkeypatch.setattr(client, '_emit_stream_event', failed.append)
        client._run_message_stream(user_text, 'failed-worker')
        assert [event['type'] for event in failed] == ['start', 'error']
        assert failed[-1]['generation_id'] == failed[0]['generation_id']
        assert 'analysis could not be completed' in failed[-1]['error']
        assert failed[-1]['outcome'] == 'failed' and failed[-1]['reply'] == ''
        assert [prompt.json_schema for prompt in model.prompts[previous_calls:]] == [
            INTENT_ONLY_SCHEMA, AUTOMATIC_FINDING_SCHEMA,
        ]
        saved = read(url, '/api/state')['messages'][-1]
        assert (saved['id'], saved['content'], saved['outcome']) == (failed[-1]['generation_id'], '', 'failed')
        assert read(url, '/api/chat', {'message': 'Recovered?', 'request_id': 'after-failure'})['reply'] == 'Direct reply.'
    logged = capsys.readouterr().out
    assert 'completed-findings' in logged
    assert 'R17' not in logged and finding not in logged and 'Choose the blue crate.' not in logged
    assert model.closed


def seed(companion):
    store = companion._ensure_store()
    session = companion._session("popup")
    for index in range(4):
        turn = uuid4()
        store.append_turn(session.conversation_id,
                          HistoryEntry(turn, "user", f"Original dialogue {index}. " + "old note " * 28),
                          HistoryEntry(turn, "assistant", "Tentative reply. " * 15))


@pytest.mark.parametrize("trigger", ["periodic", "pressure"])
def test_production_adopts_valid_segments_for_both_triggers(runtime, trigger):
    _, models = runtime
    interval = .03 if trigger == "periodic" else 300
    companion = next_adapter.NextCompanion(policy=AutomaticContextPolicy(
        segment_exchanges=4, periodic_interval_seconds=interval))
    seed(companion)
    with listener(companion) as url:
        if trigger == "pressure":
            turn = read(url, "/api/chat", {"message": "Long fixture " + "word " * 900})
            assert turn["outcome"] == "completed"
        manager = companion._host.automatic
        with manager._condition:
            assert manager._condition.wait_for(
                lambda: manager.status(companion._host.session.conversation_id).state == "completed", 5)
        status = read(url, "/api/context")["context"]
        assert status["maintenance"]["trigger"] == trigger
        assert status["periodic"]["interval_seconds"] == interval
        assert len([prompt for prompt in models[0].prompts if prompt.json_schema]) == 1
        read(url, "/api/chat", {"message": "Read the older fixture."})
        assert read(url, "/api/context")["context"]["included_segments"] == ["s1"]
        assert read(url, "/api/state")["messages"][0]["content"].startswith("Original dialogue 0.")


def test_model_and_data_failures_do_not_fall_back_or_leave_an_owner(runtime, monkeypatch, tmp_path):
    path, models = runtime
    factory = next_adapter.NativeModel
    for failure in (FileNotFoundError("Missing model"), ModuleNotFoundError("Missing native binding"),
                    RuntimeError("Native initialization failed")):
        def load(_path):
            raise failure
        monkeypatch.setattr(next_adapter, "NativeModel", load)
        companion = next_adapter.NextCompanion()
        try:
            with pytest.raises(type(failure), match=str(failure)):
                companion.start()
            assert not companion.ready
        finally:
            companion.close()
    monkeypatch.setattr(next_adapter, "NativeModel", factory)
    bad = tmp_path / "not-a-directory"
    bad.write_text("invalid state path")
    companion = next_adapter.NextCompanion()
    companion.root = bad
    with pytest.raises(OSError):
        companion.start()
    companion.close()
    state = next(path.glob("profiles/*/sessions/*.json"))
    state.write_text("{broken fixture}")
    companion = next_adapter.NextCompanion()
    try:
        with pytest.raises(ValueError):
            companion.start()
        assert not models
    finally:
        companion.close()


def test_second_listener_and_store_owner_cannot_load_duplicate_model(runtime):
    _, models = runtime
    with listener(next_adapter.NextCompanion()) as url:
        port = int(url.rsplit(":", 1)[1])
        with pytest.raises(OSError):
            server.BackgroundServer("127.0.0.1", port)
        duplicate = next_adapter.NextCompanion()
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                duplicate.start()
        finally:
            duplicate.close()
        assert len(models) == 1


def test_existing_state_snapshot_reads_archive_once(runtime):
    companion = next_adapter.NextCompanion()
    try:
        seed(companion)
        store = companion._ensure_store()
        with patch.object(store, "_load", wraps=store._load) as loads:
            snapshot = companion.snapshot()
        assert snapshot["revision"] == 4 and len(snapshot["turns"]) == 8
        assert loads.call_count == 1
    finally:
        companion.close()


def test_relocation_reopens_phase6c_profile_without_changing_original_data(runtime):
    path, models = runtime
    source = Path(__file__).resolve().parents[2] / "docs/step6c-native/store"
    originals = {file.relative_to(source): file.read_bytes() for file in source.rglob("*.json")}
    shutil.copytree(source, path)
    assert next_adapter.NextCompanion().root == path
    with listener(next_adapter.NextCompanion()) as url:
        state = read(url, "/api/state")
        assert state["version"] == 11
        assert any(message["outcome"] == "cancelled" for message in state["messages"])
        memory, = read(url, "/api/memory")["facts"]
        assert memory["content"] == "The blue box label is Alder."
        assert read(url, "/api/context")["context"]["periodic"]["interval_seconds"] == 300
        assert {file.relative_to(path): file.read_bytes() for file in path.rglob("*.json")} == originals
        result = read(url, "/api/chat", {"message": "Review the fictional shelf location."})
        assert result["messages"][:-2] == state["messages"]
        context = read(url, "/api/context")["context"]
        assert context["included_segments"] == ["s1"]
        assert context["selected_memory_ids"] == [memory["id"]]
        assert models[0].prompts[-1].text.count("The blue box label is Alder.") == 1
    assert {file.relative_to(source): file.read_bytes() for file in source.rglob("*.json")} == originals


def test_shutdown_cancels_periodic_job_before_model_close(runtime, monkeypatch):
    _, models = runtime
    entered, release = Event(), Event()
    original = Model.stream

    def blocked(self, prompt, *, cancellation):
        if prompt.json_schema:
            entered.set()
            assert release.wait(5)
            cancellation.raise_if_cancelled()
        yield from original(self, prompt, cancellation=cancellation)

    monkeypatch.setattr(Model, "stream", blocked)
    companion = next_adapter.NextCompanion(policy=AutomaticContextPolicy(
        segment_exchanges=4, periodic_interval_seconds=.03))
    seed(companion)
    with listener(companion):
        manager = companion._host.automatic
        assert entered.wait(3)
        job = companion._scheduler.get_job(next_adapter._PROFILE_ID, manager.last_operation.job_id)
        thread = Thread(target=companion.close)
        thread.start()
        try:
            assert job.cancellation._signal.wait(3)
            assert not models[0].closed
        finally:
            release.set()
            thread.join(5)
        assert not thread.is_alive() and models[0].closed
        assert not manager._thread.is_alive()


def test_legacy_audit_preserves_records_and_never_approves_memories(tmp_path):
    audit_file = Path(__file__).resolve().parents[2] / "scripts/audit_v2_history.py"
    audit = runpy.run_path(str(audit_file))["audit"]
    state = {"schema_version": 20, "profiles": {"old": {"conversations": {
        "good": {"turns": [
            {"id": "user", "role": "user", "content": "Real fixture statement.", "created_at": 1},
            {"id": "assistant", "role": "assistant", "content": "Fixture reply.", "created_at": 2},
        ]}, "incomplete": {"turns": [{"role": "user", "content": "Keep me."}]},
    }, "memories": [{"title": "unsupported draft"}, {"confirmed": True}]}}}
    path = tmp_path / "v2.json"
    raw = json.dumps(state, indent=3).encode()
    path.write_bytes(raw)
    result = audit(path)
    assert result["pairs_without_outcome"] == 1
    assert result["incompatible_sessions"] == 1 and result["memories_withheld"] == 2
    assert path.read_bytes() == raw and list(tmp_path.iterdir()) == [path]
