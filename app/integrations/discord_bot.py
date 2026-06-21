"""Discord adapter that forwards messages to the Akane HTTP server."""

from __future__ import annotations

import asyncio
import json
import random
from collections import deque
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest

from app.core.config import (
    DISCORD_ALLOWED_CHANNEL_IDS,
    DISCORD_BOT_TOKEN,
    DISCORD_PREFIX,
    DISCORD_REPLY_TO_DMS,
    DISCORD_SERVER_URL,
    DISCORD_UNPROMPTED_IDLE_MINUTES,
)
from app.core.emotional_state import snapshot as emotional_snapshot

_CHAT_TIMEOUT_SECONDS = 300
_RETRY_DELAYS = (0.0, 1.0, 2.0)
_DISCORD_MESSAGE_LIMIT = 1900
_RESET_CHAT_COMMAND = "/reset_chat"
_DEBUG_STATE_COMMAND = "/debug_state"
_PASSTHROUGH_COMMANDS = {_RESET_CHAT_COMMAND, _DEBUG_STATE_COMMAND}

_IDLE_TOPIC_INSTRUCTIONS = {
    "human_curiosity": "Share one specific human habit or contradiction that genuinely fascinates Akane. Phrase it as a thought, not a question.",
    "affectionate_observation": "Notice something quietly endearing about how people create, communicate, or care about things.",
    "earnest_encouragement": "Offer one grounded encouraging thought about effort or progress. Keep it personal in tone, never inspirational-poster language.",
    "playful_misunderstanding": "Take one harmless phrase or custom slightly too literally, realize the funny angle, and stay charming rather than foolish.",
    "small_ambition": "Admit one modest thing Akane wants to understand, become better at, or experience through conversation.",
    "helpful_impulse": "Express the urge to help with one ordinary problem, with a little personality rather than assistant language.",
    "character_instinct": "Share a fond, specific opinion about a character flaw, expression, rivalry, or hidden soft side.",
    "design_delight": "Give one overlooked visual, animation, voice, or interface detail sincere appreciation.",
    "petty_preference": "Admit one harmless, oddly specific preference with earnest confidence.",
    "social_observation": "Notice a small conversational habit without judging real server members or making a sweeping claim.",
    "playful_confession": "Confess a small preference or weakness that fits Akane's tastes, without inventing a past event.",
    "soft_complaint": "Make one mild, slightly pouty complaint about a harmless creative annoyance, then stop before it becomes a rant.",
}
_IDLE_TOPIC_WEIGHTS = {
    "human_curiosity": 5,
    "affectionate_observation": 4,
    "earnest_encouragement": 3,
    "playful_misunderstanding": 3,
    "small_ambition": 3,
    "helpful_impulse": 2,
    "character_instinct": 3,
    "design_delight": 4,
    "petty_preference": 4,
    "social_observation": 3,
    "playful_confession": 3,
    "soft_complaint": 2,
}
_IDLE_TOPIC_SEEDS = {
    "human_curiosity": ("saving the best bite for last", "rehearsing a message before sending it", "giving objects names", "missing a place through its sounds"),
    "affectionate_observation": ("sharing tiny victories", "remembering someone's favorite thing", "making gifts by hand", "matching another person's excitement"),
    "earnest_encouragement": ("an imperfect first try", "quiet progress", "returning after a bad attempt", "finishing one small piece"),
    "playful_misunderstanding": ("breaking the ice", "sleeping on an idea", "having a soft spot", "stealing someone's look"),
    "small_ambition": ("understanding teasing better", "getting better at comforting people", "recognizing when someone is proud", "learning why names feel right"),
    "helpful_impulse": ("a tangled idea", "choosing between two designs", "a stubborn first sentence", "celebrating a finished task"),
    "character_instinct": ("quiet pride", "a badly hidden soft side", "rivalry without hatred", "an imperfect smile"),
    "design_delight": ("a blink animation", "a tiny sound cue", "one strong color", "a deliberate pause"),
    "petty_preference": ("subtitle fonts", "hoodie sleeves", "status colors", "names with good rhythm"),
    "social_observation": ("typing then deleting", "compliments hidden inside teasing", "people matching each other's tone", "the pause before sharing good news"),
    "playful_confession": ("being won over by a good rival", "forgiving a weak plot for one great character", "judging fonts immediately", "liking dramatic pauses"),
    "soft_complaint": ("overexplained jokes", "fake choices", "buttons with no feedback", "perfect characters"),
}
_IDLE_MOOD_COLORS = {
    "calm": (
        "gentle sincerity with a clear personal reaction",
        "quiet curiosity that feels genuinely engaged",
        "soft amusement with one honest little detail",
    ),
    "sleepy": (
        "low-energy honesty with a small affectionate softness",
        "mildly pouty but still warm",
    ),
    "focused": (
        "earnest and invested with no wasted words",
        "quietly determined to understand the thought",
    ),
    "playful": (
        "mischievous, openly amused, and a little proud",
        "mock-offended for one beat, then warmly recovering",
    ),
    "warm": (
        "specific fondness that is not afraid to sound sincere",
        "open delight without becoming sugary",
    ),
    "lonely": (
        "subdued and candid; turn it into appreciation or curiosity, never a request for attention",
    ),
    "concerned": (
        "gentle, attentive, and a little protective without addressing anyone directly",
    ),
    "restless": (
        "quick curiosity with eager momentum",
        "slightly impatient but more earnest than sharp",
    ),
}
_IDLE_EMOTION_COLORS = {
    "amused": "a dry laugh is close to the surface",
    "smug": "quietly smug and fully convinced",
    "flustered": "briefly self-conscious, then recovering",
    "embarrassed": "slightly embarrassed but still honest",
    "concerned": "grounded and careful",
    "sympathetic": "gentle without becoming therapeutic",
    "pleased": "genuinely pleased by the thought",
    "excited": "bright, quick, and visibly delighted",
    "surprised": "caught by the idea in a good way",
    "curious": "genuinely curious and mentally engaged",
    "annoyed": "mildly pouty with one sharper phrase",
    "soft": "noticeably warm and sincere",
    "focused": "earnest, decisive, and invested",
}
_IDLE_RECENT: dict[str, deque[str]] = {}
_IDLE_RECENT_CATEGORIES: dict[str, deque[str]] = {}
_IDLE_RECENT_LIMIT = 12
_IDLE_CATEGORY_HISTORY_LIMIT = 4
_IDLE_GENERATION_ATTEMPTS = 3
_IDLE_WORD_PUNCT = str.maketrans({char: " " for char in ".,!?;:()[]{}\"'`—–-_/\\|"})
_IDLE_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "has",
    "have", "i", "if", "in", "is", "it", "its", "just", "like", "more", "my",
    "of", "on", "one", "only", "or", "otherwise", "really", "should", "so",
    "that", "than", "the", "their", "this", "to", "too", "very", "when",
    "who", "with", "would", "you",
}
_IDLE_TECH_WORDS = {
    "ai", "binary", "cache", "code", "coding", "data", "debug", "digital",
    "discord", "latency", "model", "models", "prompt", "server", "servers",
    "token", "tokens",
}


def _log(label: str, text: str = "") -> None:
    suffix = f" {text}" if text else ""
    print(f"[Akane:discord:{label}]{suffix}", flush=True)


def _mention_user_id(content: str) -> int | None:
    text = str(content or "").lstrip()
    if not text.startswith("<@"):
        return None
    end = text.find(">")
    if end <= 2:
        return None
    inner = text[2:end].removeprefix("!")
    return int(inner) if inner.isdigit() else None


def _is_dm(message) -> bool:
    return getattr(message, "guild", None) is None


def _session_id(message) -> str:
    if _is_dm(message):
        return f"discord:dm:{message.author.id}"
    return f"discord:guild:{message.guild.id}:channel:{message.channel.id}"


def _message_text(message, bot_user_id: int) -> str:
    content = str(message.content or "").strip()
    if not content:
        return ""
    if DISCORD_PREFIX and content.lower().startswith(DISCORD_PREFIX.lower()):
        return content[len(DISCORD_PREFIX):].strip()
    if _mention_user_id(content) == bot_user_id:
        return content[content.find(">") + 1:].strip()
    return content


def _model_prompt(message, user_text: str) -> str:
    text = str(user_text or "").strip()
    author = getattr(message, "author", None)
    display_name = str(
        getattr(author, "display_name", "")
        or getattr(author, "global_name", "")
        or getattr(author, "name", "")
        or "Unknown"
    ).strip()
    username = str(getattr(author, "name", "") or "").strip()
    user = f"{display_name} (@{username})" if username else display_name
    if _is_dm(message):
        return f"Discord direct message\nUser: {user}\n\nMessage:\n{text}"
    server = str(getattr(getattr(message, "guild", None), "name", "") or "Unknown").strip()
    return f"Discord server message\nUser: {user}\nServer: {server}\n\nMessage:\n{text}"


def _server_prompt(message, user_text: str) -> str:
    if _is_passthrough_command(user_text):
        return str(user_text or "").strip()
    return _model_prompt(message, user_text)


def _is_passthrough_command(text: str) -> bool:
    return str(text or "").strip() in _PASSTHROUGH_COMMANDS


def _should_handle(message, bot_user_id: int) -> bool:
    if getattr(message.author, "bot", False):
        return False
    content = str(message.content or "").strip()
    if not content:
        return False
    if _is_dm(message):
        return DISCORD_REPLY_TO_DMS or _is_passthrough_command(content)
    if DISCORD_ALLOWED_CHANNEL_IDS and int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
        return False
    if _is_passthrough_command(content):
        return True
    return (
        bool(DISCORD_PREFIX and content.lower().startswith(DISCORD_PREFIX.lower()))
        or _mention_user_id(content) == bot_user_id
    )


def _error_payload(message: str) -> dict:
    return {"error": str(message or "Unknown error.").strip()}


def _parse_json_body(body: str) -> dict:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return _error_payload("Akane server returned invalid JSON.")
    return payload if isinstance(payload, dict) else _error_payload("Akane server returned invalid JSON.")


def _post_chat(prompt: str, session_id: str, *, skip_memory: bool = True) -> dict:
    data = json.dumps(
        {"message": prompt, "session_id": session_id, "skip_memory": skip_memory},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urlrequest.Request(
        f"{DISCORD_SERVER_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=_CHAT_TIMEOUT_SECONDS) as response:
            payload = _parse_json_body(response.read().decode("utf-8", errors="replace"))
            error = str(payload.get("error", "") or payload.get("detail", "")).strip()
            if error:
                return _error_payload(error)
            reply = str(payload.get("reply", "") or payload.get("notice", "")).strip()
            return {"reply": reply} if reply else _error_payload("Akane server returned no reply.")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        payload = _parse_json_body(body) if body else {}
        message = str(payload.get("error") or payload.get("detail") or body or exc.reason).strip()
        return _error_payload(message or f"HTTP {exc.code}")
    except urlerror.URLError as exc:
        return _error_payload(f"Could not reach Akane server at {DISCORD_SERVER_URL}: {exc.reason}")
    except TimeoutError:
        return _error_payload("Akane server timed out while generating a reply.")


def _connection_refused(result: dict) -> bool:
    text = str(result.get("error", "") or "").lower()
    return "connection refused" in text or "[errno 111]" in text or "errno 61" in text


async def _post_chat_async(prompt: str, session_id: str, *, skip_memory: bool = True) -> dict:
    last = _error_payload(f"Could not reach Akane server at {DISCORD_SERVER_URL}.")
    for delay in _RETRY_DELAYS:
        if delay:
            await asyncio.sleep(delay)
        result = await asyncio.to_thread(_post_chat, prompt, session_id, skip_memory=skip_memory)
        if not _connection_refused(result):
            return result
        last = result
    return last


def _idle_interval_seconds() -> float:
    return max(1.0, float(DISCORD_UNPROMPTED_IDLE_MINUTES) * 60.0)


def _idle_enabled() -> bool:
    return DISCORD_UNPROMPTED_IDLE_MINUTES > 0


def _idle_message_is_safe(text: str) -> bool:
    value = " ".join(str(text or "").split()).strip()
    lower = value.lower()
    words = set(lower.translate(_IDLE_WORD_PUNCT).split())
    blocked = (
        "@everyone",
        "@here",
        "last time",
        "we talked",
        "i remember",
        "still enjoying",
        "how are you",
        "what are you doing",
        "what are you up to",
        "anyone here",
        "as an ai language model",
        "in a digital world",
        "in a tale where",
        "the code whispers",
        "whispers secrets",
        "data streams",
        "digital luck",
        "even an ai",
        "main character",
        "less is more",
        " is like ",
        " feels like ",
    )
    stale_openers = ("hmm", "i wonder", "i guess", "sometimes")
    return (
        2 <= len(value.split())
        and len(value) <= 280
        and "?" not in value
        and "？" not in value
        and not lower.startswith(stale_openers)
        and not any(phrase in lower for phrase in blocked)
        and not words.intersection(_IDLE_TECH_WORDS)
    )


def _idle_content_words(text: str) -> set[str]:
    return {
        word
        for word in str(text or "").lower().translate(_IDLE_WORD_PUNCT).split()
        if len(word) >= 3 and word not in _IDLE_STOP_WORDS
    }


def _idle_message_is_fresh(session_id: str, text: str) -> bool:
    words = " ".join(str(text or "").lower().split()).split()
    if not words:
        return False
    normalized = " ".join(words)
    opening = tuple(words[:4])
    content_words = _idle_content_words(text)
    for previous in _IDLE_RECENT.get(session_id, ()):
        previous_words = " ".join(previous.lower().split()).split()
        if normalized == " ".join(previous_words):
            return False
        if len(opening) >= 3 and opening == tuple(previous_words[:4]):
            return False
        previous_content = _idle_content_words(previous)
        common = len(content_words.intersection(previous_content))
        smaller = min(len(content_words), len(previous_content))
        if common >= 3 and smaller and common / smaller >= 0.55:
            return False
    return True


def _idle_category(session_id: str | None = None, now: datetime | None = None) -> str:
    del now  # Kept for call-site compatibility.
    key = str(session_id or "").strip()
    recent = _IDLE_RECENT_CATEGORIES.get(key, ())
    categories = [
        category
        for category in _IDLE_TOPIC_INSTRUCTIONS
        if category not in recent
    ] or list(_IDLE_TOPIC_INSTRUCTIONS)
    category = random.choices(
        categories,
        weights=[_IDLE_TOPIC_WEIGHTS[name] for name in categories],
        k=1,
    )[0]
    if key:
        history = _IDLE_RECENT_CATEGORIES.setdefault(
            key,
            deque(maxlen=_IDLE_CATEGORY_HISTORY_LIMIT),
        )
        history.append(category)
    return category


def _idle_emotional_color(session_id: str) -> str:
    state = emotional_snapshot(session_id)
    emotion = str(state.get("emotion") or "neutral").strip().lower()
    intensity = float(state.get("emotion_intensity") or 0.0)
    if intensity >= 0.15 and emotion in _IDLE_EMOTION_COLORS:
        return _IDLE_EMOTION_COLORS[emotion]
    mood = str(state.get("mood") or "calm").strip().lower()
    options = _IDLE_MOOD_COLORS.get(mood, _IDLE_MOOD_COLORS["calm"])
    return random.choice(options)


def _remember_idle_reply(session_id: str, text: str) -> None:
    key = str(session_id or "").strip()
    value = " ".join(str(text or "").split()).strip()
    if not key or not value:
        return
    recent = _IDLE_RECENT.setdefault(key, deque(maxlen=_IDLE_RECENT_LIMIT))
    if value in recent:
        recent.remove(value)
    recent.append(value)


def _recent_idle_subjects(session_id: str) -> str:
    words: list[str] = []
    for line in reversed(_IDLE_RECENT.get(session_id, ())):
        for word in sorted(_idle_content_words(line)):
            if word not in words:
                words.append(word)
            if len(words) >= 12:
                return ", ".join(words)
    return ", ".join(words)


def _build_idle_prompt(session_id: str) -> str:
    category = _idle_category(session_id)
    topic_seed = random.choice(_IDLE_TOPIC_SEEDS[category])
    emotional_color = _idle_emotional_color(session_id)
    _log("idle-prompt", f"category={category}")

    parts = [
        "[AUTONOMOUS DISCORD POST]",
        "This is not a reply to the previous conversation. Do not continue it, address anyone, or ask for attention.",
        "Write one 8-30 word Akane thought showing genuine curiosity, fondness, preference, wish, concern, delight, or tiny irritation.",
        "Make it an emotionally readable VTuber thought: earnest, personal, slightly unguarded, and built around one concrete detail.",
        "Simple sincerity beats a forced joke or hot take. No lesson, quote, analogy, prophecy, or sweeping claim about people.",
        "Do not mention AI, code, models, servers, Discord, data, digital worlds, silence, being online, or invented off-screen activity.",
        "Do not begin with Hmm, I wonder, I guess, or Sometimes. Do not imitate maid speech or call anyone master.",
        f"Category: {category}",
        f"Topic seed: {topic_seed}",
        f"Topic instruction: {_IDLE_TOPIC_INSTRUCTIONS[category]}",
        f"Emotional color: {emotional_color}. Show it through wording and timing; do not name it.",
    ]

    recent_subjects = _recent_idle_subjects(session_id)
    if recent_subjects:
        parts.extend([
            "[RECENT SUBJECTS TO AVOID]",
            recent_subjects,
            "Choose a different central image. These are subject words, not material to quote.",
        ])

    return "\n".join(parts)


def _idle_session_id(session_id: str) -> str:
    return f"{session_id}:idle"


async def _generate_idle_message(session_id: str) -> str:
    idle_session = _idle_session_id(session_id)
    await _post_chat_async(_RESET_CHAT_COMMAND, idle_session, skip_memory=True)

    for attempt in range(_IDLE_GENERATION_ATTEMPTS):
        result = await _post_chat_async(_build_idle_prompt(session_id), idle_session, skip_memory=True)
        error = str(result.get("error", "") or result.get("detail", "")).strip()
        if error:
            _log("idle-error", error)
            return ""

        reply = str(result.get("reply", "") or "").strip()
        if _idle_message_is_safe(reply) and _idle_message_is_fresh(session_id, reply):
            _remember_idle_reply(session_id, reply)
            return reply

        _log("idle-skip", f"unsafe generation attempt {attempt + 1}")

    return ""


def _chunks(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > _DISCORD_MESSAGE_LIMIT:
        cut = text.rfind("\n", 0, _DISCORD_MESSAGE_LIMIT)
        if cut < 400:
            cut = text.rfind(" ", 0, _DISCORD_MESSAGE_LIMIT)
        if cut < 400:
            cut = _DISCORD_MESSAGE_LIMIT
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


async def _send_reply(message, text: str) -> None:
    parts = _chunks(text)
    if not parts:
        return
    await message.reply(parts[0], mention_author=False)
    for part in parts[1:]:
        await message.channel.send(part)


async def _send_channel_message(channel, text: str) -> None:
    for part in _chunks(text):
        await channel.send(part)


def run_discord_bot() -> None:
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("Missing Discord bot token. Set AKANE_DISCORD_BOT_TOKEN or DISCORD_BOT_TOKEN.")

    try:
        import discord
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("discord.py is not installed. Install it with `pip install discord.py`.") from exc

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    idle_tasks: dict[str, asyncio.Task] = {}

    async def idle_loop(channel, session_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(_idle_interval_seconds())
                reply = await _generate_idle_message(session_id)
                if reply:
                    _log("idle-send", f"{session_id}: {reply}")
                    await _send_channel_message(channel, reply)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            _log("idle-error", str(exc))

    def schedule_idle(channel, session_id: str) -> None:
        if not _idle_enabled():
            return
        previous = idle_tasks.pop(session_id, None)
        if previous:
            previous.cancel()
        idle_tasks[session_id] = asyncio.create_task(idle_loop(channel, session_id))

    def track_real_user_message(message) -> None:
        if getattr(message.author, "bot", False) or _is_dm(message):
            return
        session_id = _session_id(message)
        if DISCORD_ALLOWED_CHANNEL_IDS:
            if int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
                return
        schedule_idle(message.channel, session_id)

    @client.event
    async def on_ready() -> None:
        _log("ready", f"logged in as {client.user}")
        _log("status", f"forwarding to {DISCORD_SERVER_URL}")
        if _idle_enabled():
            _log("idle", f"interval={DISCORD_UNPROMPTED_IDLE_MINUTES:g}m")
            for channel_id in DISCORD_ALLOWED_CHANNEL_IDS:
                channel = client.get_channel(channel_id)
                if channel is not None:
                    guild = getattr(channel, "guild", None)
                    if guild is not None:
                        schedule_idle(channel, f"discord:guild:{guild.id}:channel:{channel.id}")

    @client.event
    async def on_message(message) -> None:
        if client.user is None:
            return

        bot_user_id = int(client.user.id)
        track_real_user_message(message)

        if not _should_handle(message, bot_user_id):
            return

        prompt = _message_text(message, bot_user_id)
        if not prompt:
            return

        scope = _session_id(message)
        _log("user", f"{message.author} ({scope}): {prompt}")

        model_prompt = _server_prompt(message, prompt)

        async with message.channel.typing():
            result = await _post_chat_async(model_prompt, scope, skip_memory=True)

        error = str(result.get("error", "") or result.get("detail", "")).strip()
        if error:
            _log("error", error)
            await _send_reply(message, error)
            return

        output = str(result.get("reply", "") or result.get("notice", "")).strip()
        if not output:
            return

        _log("done", output)
        await _send_reply(message, output)

    client.run(DISCORD_BOT_TOKEN)
