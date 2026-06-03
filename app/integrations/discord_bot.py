"""Discord adapter that forwards messages to the Akane HTTP server."""

from __future__ import annotations

import asyncio
import json
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

_CHAT_TIMEOUT_SECONDS = 300
_RETRY_DELAYS = (0.0, 1.0, 2.0)
_DISCORD_MESSAGE_LIMIT = 1900
_RESET_CHAT_COMMAND = "/reset_chat"
_IDLE_SYNTHETIC_MESSAGE = "Write one short autonomous Discord message from Akane."


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
    return getattr(message.guild, "id", None) is None


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


def _one_line(value, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].strip()


def _author_name(author) -> str:
    display = _one_line(getattr(author, "display_name", ""))
    global_name = _one_line(getattr(author, "global_name", ""))
    username = _one_line(getattr(author, "name", ""))
    if display and username and display != username:
        return f"{display} (@{username})"
    if global_name and username and global_name != username:
        return f"{global_name} (@{username})"
    return display or username or "Unknown user"


def _model_prompt(message, user_text: str) -> str:
    author = getattr(message, "author", None)
    guild = getattr(message, "guild", None)
    if guild is not None:
        lines = [
            "Discord server message",
            f"User: {_author_name(author)}",
            f"Server: {_one_line(getattr(guild, 'name', ''))}",
        ]
    else:
        lines = [
            "Discord direct message",
            f"User: {_author_name(author)}",
        ]
    lines.extend(("", "Message:", str(user_text or "").strip()))
    return "\n".join(lines).strip()


def _server_prompt(message, user_text: str) -> str:
    text = str(user_text or "").strip()
    return text if text == _RESET_CHAT_COMMAND else _model_prompt(message, text)


def _is_reset_chat_text(text: str) -> bool:
    return str(text or "").strip() == _RESET_CHAT_COMMAND


def _should_handle(message, bot_user_id: int) -> bool:
    if getattr(message.author, "bot", False):
        return False
    content = str(message.content or "").strip()
    if not content:
        return False
    if _is_dm(message):
        return DISCORD_REPLY_TO_DMS or _is_reset_chat_text(content)
    if DISCORD_ALLOWED_CHANNEL_IDS and int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
        return False
    if _is_reset_chat_text(content):
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
    return len(value.split()) >= 2


async def _generate_idle_message(session_id: str) -> str:
    for attempt in range(2):
        result = await _post_chat_async(_IDLE_SYNTHETIC_MESSAGE, session_id, skip_memory=True)
        error = str(result.get("error", "") or result.get("detail", "")).strip()
        if error:
            _log("idle-error", error)
            return ""
        reply = str(result.get("reply", "") or "").strip()
        if _idle_message_is_safe(reply):
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
