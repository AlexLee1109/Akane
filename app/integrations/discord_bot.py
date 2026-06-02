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
)

_CHAT_TIMEOUT_SECONDS = 300
_RETRY_DELAYS = (0.0, 1.0, 2.0)
_DISCORD_MESSAGE_LIMIT = 1900


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


def _should_handle(message, bot_user_id: int) -> bool:
    if getattr(message.author, "bot", False):
        return False
    if _is_dm(message):
        return DISCORD_REPLY_TO_DMS
    if DISCORD_ALLOWED_CHANNEL_IDS and int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
        return False
    content = str(message.content or "").strip()
    if not content:
        return False
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


def _parse_stream_line(line: str) -> tuple[str, str]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(event, dict):
        return "", ""
    event_type = str(event.get("type", "") or "")
    if event_type == "error":
        return "error", str(event.get("error") or event.get("detail") or "Streaming failed.").strip()
    if event_type == "done":
        return "reply", str(event.get("reply", "") or "").strip()
    if event_type == "delta":
        return "delta", str(event.get("content", "") or "").strip()
    return "", ""


def _post_chat(prompt: str, session_id: str, *, skip_memory: bool = True) -> dict:
    data = json.dumps(
        {"message": prompt, "session_id": session_id, "skip_memory": skip_memory},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urlrequest.Request(
        f"{DISCORD_SERVER_URL}/api/chat/stream",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=_CHAT_TIMEOUT_SECONDS) as response:
            content_type = str(response.headers.get("Content-Type", "") or "").lower()
            if "application/json" in content_type:
                return _parse_json_body(response.read().decode("utf-8", errors="replace"))

            latest_delta = ""
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                kind, value = _parse_stream_line(line)
                if kind == "error":
                    return _error_payload(value)
                if kind == "reply" and value:
                    return {"reply": value}
                if kind == "delta" and value:
                    latest_delta = value
            return {"reply": latest_delta} if latest_delta else _error_payload("Akane server returned no reply.")
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

    @client.event
    async def on_ready() -> None:
        _log("ready", f"logged in as {client.user}")
        _log("status", f"forwarding to {DISCORD_SERVER_URL}")

    @client.event
    async def on_message(message) -> None:
        if client.user is None:
            return
        bot_user_id = int(client.user.id)
        if not _should_handle(message, bot_user_id):
            return

        prompt = _message_text(message, bot_user_id)
        if not prompt:
            return

        scope = _session_id(message)
        _log("user", f"{message.author} ({scope}): {prompt}")
        model_prompt = _model_prompt(message, prompt)

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
