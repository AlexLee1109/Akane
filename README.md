# Akane

Akane is a local-first AI companion built around one persistent character, one
canonical state store, and one local language-model runtime. The desktop popup,
Discord adapter, private web chat, public demo, and optional VS Code bridge are
interfaces around the same conversation core.

Akane currently communicates through text. Speech, physical sensing, Live2D
control, and avatar animation are not implemented. Those are possible external
interfaces; her established visible form is already her body and does not depend
on a renderer or sensor.

## What v2 owns

- `Character` owns Akane's compact starting identity and seed interests.
- `SelfItem` owns developed curiosities, interests, preferences, opinions,
  goals, and tendencies, including strength, confidence, reasons, lifecycle,
  and provenance.
- `Memory` stores only grounded user, Akane, or shared facts and events.
- `Mood` is the one decaying emotional state.
- `Relationship` is the one per-profile relationship record.
- `InnerLife` stores thoughts that may continue between conversations.
- `Store` is the only persistence authority, and one canonical JSON document
  is the only source of truth.
- `InferenceRuntime` serializes llama.cpp inference and prioritizes visible work
  over reflection and autonomy.

The visible dialogue model writes only Akane's reply. A separate, low-priority
reflection pass may later propose memory, Self, mood, or relationship changes.
Deterministic validation checks ownership, exact source provenance, ranges,
references, lifecycle direction, and canonical shape before `Store.commit()`
applies a transaction. Natural-language grounding remains a prompt/model task,
not a keyword classifier.

## Requirements

- Python 3.10 or newer
- An instruction-tuned GGUF with an embedded chat template
- `llama-cpp-python` built for the target machine

The default model path is:

```text
models/gemma-4-E4B-it-Q4_K_M.gguf
```

Override it with `AKANE_MODEL_PATH`, or copy
`app/secrets/local_secrets.py.example` to `app/secrets/local_secrets.py`.
Never commit the populated secrets file.

## Running Akane

```bash
python -m app [server|popup|discord]
```

Server and web chat:

```bash
python -m app server
```

Desktop popup:

```bash
python -m app popup
```

Discord adapter:

```bash
AKANE_DISCORD_BOT_TOKEN=... python -m app discord
```

The Discord process is a thin HTTP adapter and does not load a second model or
maintain companion state. Start the standalone server first when popup and
Discord should share one process.

The permanent static website is hosted at
`https://alexlee1109.github.io/Akane/`. See
[public demo deployment](docs/public-demo-deployment.md) for Raspberry Pi,
ngrok, and GitHub Pages configuration.

## Conversation flow

```text
interface input
→ authorized profile and conversation
→ one immutable context snapshot
→ optional private deliberation for a hard turn
→ compact dialogue prompt
→ foreground model reservation
→ caller-selected complete reply or token stream
→ atomic completed-turn commit
→ mark or extend one coalesced reflection range
```

Popup and web clients retain incremental streaming. Discord uses the complete
JSON chat endpoint and sends the finished reply without placeholder messages or
message edits.

Recent raw dialogue has priority over retrieved history. The context builder
selects only relevant Self items, memories, mood, relationship notes, active
thoughts, time, and bounded context supplied by connected integrations.

Reflection is deliberately outside visible generation and waits for the configured
foreground idle grace period:

```text
bounded batch of unreflected exchanges
→ one compact reflection JSON proposal
→ deterministic validation
→ Store.commit()
→ one atomic JSON replacement
```

Most reflection proposals should be empty or small. A user statement about
Akane does not become Akane's preference; Self evidence must come from what
Akane chose to express. Subjective choices are allowed without fake physical
history.

## InnerLife and autonomy

One low-duty-cycle coordinator performs at most one model call per wake: one
idle, coalesced reflection first, otherwise one due InnerLife tick. InnerLife can
remain quiet, continue an existing thought, form a
related curiosity, or create/revise/complete/abandon a lightweight goal through
the canonical Self. It cannot fabricate external activities.

Only an important, active, explicitly share-worthy thought can enter the
proactive delivery queue. Popup and Discord delivery use legacy endpoint names
for wire compatibility, but do not contain a second initiative system.

Foreground reservations outrank reflection and autonomy. Optional background
failures are recorded and retried without invalidating conversation history.

## Persistence

State lives in `data/akane_state.json` by default. This one schema-versioned
document contains:

```text
profiles
  self                opinions, preferences, interests, curiosities,
                      goals, tendencies, revisions
  mood                relationship            memories
  inner_life          thoughts
  conversations       turns
reflection_jobs       proactive_queue         change_log
```

`Store` copies the current document, validates the candidate, writes a temporary
file in the same directory, flushes and syncs it, and atomically replaces the
canonical file before publishing the new in-memory revision. A failed load or
write never silently resets state. Set `AKANE_DATA_DIR` to move the state file.
Legacy split state files are not read or recreated.

## Commands

- `/reset_chat` resets only the active recent conversation.
- `/clear` does the same with a shorter notice.
- `/forget_me` removes the active profile's conversation and durable state.
- `/debug_state` shows a compact state, selection, prompt-size, timing, and
  revision snapshot without exposing prompt contents.

## Configuration

All settings are loaded in `app/core/config.py` from `AKANE_*` environment
variables or the optional local secrets file.

| Area | Important settings |
| --- | --- |
| Server | `AKANE_SERVER_HOST`, `AKANE_SERVER_PORT`, `AKANE_SERVER_API_TOKEN` |
| Public API | `AKANE_PUBLIC_API_ENABLED`, `AKANE_PUBLIC_ALLOWED_ORIGINS`, guest limits and timeouts |
| Clients | `AKANE_POPUP_BACKEND_URL`, `AKANE_DISCORD_SERVER_URL` |
| Model | `AKANE_MODEL_PATH`, `AKANE_LLAMA_CONTEXT_WINDOW`, batch, thread, and offload settings |
| Generation | `AKANE_MAX_TOKENS`, sampling settings, queue size, queue timeout |
| State | `AKANE_DATA_DIR`, recent-turn and retrieval limits |
| Background | `AKANE_AUTONOMY_INTERVAL_SECONDS`, `AKANE_BACKGROUND_IDLE_GRACE_SECONDS`, reflection/InnerLife token limits and intervals |
| Diagnostics | `AKANE_PROMPT_DEBUG`, `AKANE_TIMING` |

With `AKANE_PROMPT_DEBUG=1`, startup reports the resolved character paths,
hashes, and exact model-token counts. The first dialogue turn in that process
also logs one exact rendered chat-template prompt, its SHA-256, and section
token counts. This development output contains conversation text; do not enable
it for untrusted or secret-bearing traffic. Restart the process to capture a
different requested turn.

Public mode binds to loopback and requires exact HTTPS CORS origins outside
localhost. Public clients receive isolated, expiring `public:guest:<uuid>`
profiles and can never select `local:owner`.

## Repository structure

```text
app/core/
  state.py           canonical value objects and proposals
  store.py           canonical JSON validation and atomic persistence
  character.py       static starting foundation
  mind.py            deterministic psychological-state operations
  context.py         state selection and model-context presentation
  prompt.py          prompt-only compilation
  reflection.py      post-turn extraction and validation
  session.py         foreground turn and deliberation orchestration
  autonomy.py        background reflection, InnerLife, and delivery coordinator
  inference.py       shared priority-aware llama.cpp runtime
  config.py          typed centralized configuration
  utils.py           shared text and profile helpers
```