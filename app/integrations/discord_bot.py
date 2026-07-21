"""Discord adapter for Akane's normalized HTTP chat contract."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque

from app.core.config import (
    DISCORD_ALLOWED_CHANNEL_IDS,
    DISCORD_BOT_TOKEN,
    DISCORD_PREFIX,
    DISCORD_REPLY_TO_DMS,
    DISCORD_SERVER_URL,
    SERVER_API_TOKEN,
)
from app.core.utils import compact_text

_CHAT_TIMEOUT_SECONDS = 300
_UNPROMPTED_INTERVAL_SECONDS = 4 * 60 * 60  # 14,400 seconds
_DISCORD_MESSAGE_LIMIT = 1900
_RECENT_EVENT_LIMIT = 512
_RECENT_EVENT_IDS: deque[int] = deque()
_RECENT_EVENT_SET: set[int] = set()
_RAW_MENTION = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
_CUSTOM_EMOJI = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
_UNPROMPTED_PROMPT = """[UNPROMPTED DISCORD POST]
Write one short, natural, unprompted thought as Akane. Keep it one to three sentences. Continue from a supplied concrete interest, recent concern, or activity when one is relevant; use a specific thought, reaction, or opinion instead of generic atmosphere. Do not pretend this is a reply or address a specific person.
Do not mention timers, schedules, automation, Discord, silence, or that the chat is quiet. Do not ask for attention. Do not include @everyone, @here, a role mention, or a user mention. Return only the completed message."""


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


def _profile_id(message) -> str:
    return f"discord:user:{int(message.author.id)}"


def _conversation_id(message) -> str:
    user_id = int(message.author.id)
    if _is_dm(message):
        return f"discord:dm:{user_id}"
    return (
        f"discord:guild:{int(message.guild.id)}:"
        f"channel:{int(message.channel.id)}:user:{user_id}"
    )


def _message_text(message, bot_user_id: int) -> str:
    content = str(message.content or "").strip()
    if not content:
        return ""
    if DISCORD_PREFIX and content.lower().startswith(DISCORD_PREFIX.lower()):
        content = content[len(DISCORD_PREFIX) :].strip()
        return _normalize_discord_text(message, content)
    if _mention_user_id(content) == bot_user_id:
        content = content[content.find(">") + 1 :].strip()
    return _normalize_discord_text(message, content)


def _normalize_discord_text(message, text: str) -> str:
    value = str(text or "")
    for user in getattr(message, "mentions", ()):
        name = compact_text(getattr(user, "display_name", ""), 60) or "user"
        value = value.replace(f"<@{int(user.id)}>", f"@{name}")
        value = value.replace(f"<@!{int(user.id)}>", f"@{name}")
    for channel in getattr(message, "channel_mentions", ()):
        name = compact_text(getattr(channel, "name", ""), 60) or "channel"
        value = value.replace(f"<#{int(channel.id)}>", f"#{name}")
    for role in getattr(message, "role_mentions", ()):
        name = compact_text(getattr(role, "name", ""), 60) or "role"
        value = value.replace(f"<@&{int(role.id)}>", f"@{name}")
    value = _RAW_MENTION.sub("@mention", value)
    value = _CUSTOM_EMOJI.sub(lambda match: f":{match.group(1)}:", value)
    return value.strip()


def _display_name(message) -> str:
    author = getattr(message, "author", None)
    value = (
        getattr(author, "display_name", "")
        or getattr(author, "global_name", "")
        or getattr(author, "name", "")
    )
    return compact_text(str(value).replace("@", ""), 60)


def _reply_context(message) -> str:
    reference = getattr(message, "reference", None)
    resolved = getattr(reference, "resolved", None)
    content = compact_text(getattr(resolved, "content", ""), 420)
    if not content:
        return ""
    author = compact_text(
        getattr(getattr(resolved, "author", None), "display_name", ""),
        60,
    )
    return f"{author or 'A participant'} previously wrote: {content}"


def _should_handle(message, bot_user_id: int) -> bool:
    if getattr(message.author, "bot", False):
        return False
    content = str(message.content or "").strip()
    if not content:
        return False
    if _is_dm(message):
        return DISCORD_REPLY_TO_DMS or _is_explicit(content, bot_user_id)
    if DISCORD_ALLOWED_CHANNEL_IDS and int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
        return False
    return _is_explicit(content, bot_user_id)


def _is_explicit(content: str, bot_user_id: int) -> bool:
    return bool(
        DISCORD_PREFIX and content.lower().startswith(DISCORD_PREFIX.lower())
    ) or _mention_user_id(content) == bot_user_id


def _mark_event(message) -> bool:
    event_id = int(getattr(message, "id", 0) or 0)
    if not event_id:
        return True
    if event_id in _RECENT_EVENT_SET:
        return False
    _RECENT_EVENT_IDS.append(event_id)
    _RECENT_EVENT_SET.add(event_id)
    while len(_RECENT_EVENT_IDS) > _RECENT_EVENT_LIMIT:
        _RECENT_EVENT_SET.discard(_RECENT_EVENT_IDS.popleft())
    return True


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if SERVER_API_TOKEN:
        headers["Authorization"] = f"Bearer {SERVER_API_TOKEN}"
    return headers


def _payload(message, text: str) -> dict[str, object]:
    return {
        "message": text,
        "profile_id": _profile_id(message),
        "conversation_id": _conversation_id(message),
        "source": "discord",
        "timestamp": time.time(),
        "display_name": _display_name(message),
        "reply_context": _reply_context(message),
    }


def _unprompted_payload(channel) -> dict[str, object]:
    guild = getattr(channel, "guild", None)
    guild_id = int(getattr(guild, "id", 0) or 0)
    channel_id = int(channel.id)
    return {
        "message": _UNPROMPTED_PROMPT,
        "profile_id": "local:owner",
        "conversation_id": f"discord:guild:{guild_id}:channel:{channel_id}:autonomous",
        "source": "discord",
        "timestamp": time.time(),
        "autonomous": True,
        "skip_if_busy": True,
    }


def _valid_unprompted_message(text: str) -> bool:
    value = " ".join(str(text or "").split()).strip()
    lower = value.lower()
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", value) if part]
    blocked = (
        "@everyone",
        "@here",
        "<@",
        "timer",
        "schedule",
        "automat",
        "chat is quiet",
        "chat's quiet",
    )
    return bool(value) and len(value) <= 500 and len(sentences) <= 3 and not any(
        phrase in lower for phrase in blocked
    )


async def _post_chat(http, payload: dict[str, object]) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_CHAT_TIMEOUT_SECONDS)
    try:
        async with http.post(
            f"{DISCORD_SERVER_URL}/api/chat",
            json=payload,
            headers=_headers(),
            timeout=timeout,
        ) as response:
            text = await response.text()
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"error": "Akane server returned invalid JSON."}
            if response.status >= 400:
                return {
                    "error": data.get("error")
                    or f"Akane server returned HTTP {response.status}."
                }
            if isinstance(data, dict):
                return data
            return {"error": "Akane server returned invalid JSON."}
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return {"error": f"Could not reach the Akane server: {type(exc).__name__}."}


async def _cancel_remote(http, conversation_id: str, profile_id: str) -> None:
    try:
        async with http.post(
            f"{DISCORD_SERVER_URL}/api/chat/cancel",
            json={"conversation_id": conversation_id, "profile_id": profile_id},
            headers=_headers(),
        ) as response:
            await response.read()
    except Exception:
        pass


def _chunks(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > _DISCORD_MESSAGE_LIMIT:
        cut = text.rfind("\n\n", 0, _DISCORD_MESSAGE_LIMIT)
        if cut < 400:
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


async def _send_reply(message, text: str, allowed_mentions) -> None:
    parts = _chunks(text)
    if not parts:
        return
    await message.reply(
        parts[0],
        mention_author=False,
        allowed_mentions=allowed_mentions,
    )
    for part in parts[1:]:
        await message.channel.send(part, allowed_mentions=allowed_mentions)


def run_discord_bot() -> None:
    if not DISCORD_BOT_TOKEN:
        raise SystemExit(
            "Missing Discord bot token. Set AKANE_DISCORD_BOT_TOKEN or "
            "DISCORD_BOT_TOKEN."
        )

    try:
        import aiohttp
        import discord
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Discord support requires discord.py and aiohttp.") from exc

    class AkaneDiscordClient(discord.Client):
        http_session: aiohttp.ClientSession
        unprompted_task: asyncio.Task | None = None
        unprompted_channel = None

        async def setup_hook(self) -> None:
            self.http_session = aiohttp.ClientSession()

        async def close(self) -> None:
            if self.unprompted_task is not None:
                self.unprompted_task.cancel()
                await asyncio.gather(self.unprompted_task, return_exceptions=True)
                self.unprompted_task = None
            if hasattr(self, "http_session"):
                await self.http_session.close()
            await super().close()

    intents = discord.Intents.default()
    intents.message_content = True
    client = AkaneDiscordClient(intents=intents)
    allowed_mentions = discord.AllowedMentions.none()

    async def unprompted_loop(channel) -> None:
        while True:
            await asyncio.sleep(_UNPROMPTED_INTERVAL_SECONDS)
            try:
                result = await _post_chat(client.http_session, _unprompted_payload(channel))
                error = compact_text(result.get("error"), 240)
                if error:
                    _log("unprompted-skip", error)
                    continue
                output = str(result.get("reply") or "").strip()
                if _valid_unprompted_message(output):
                    await channel.send(output, allowed_mentions=allowed_mentions)
                    _log("unprompted-send", f"channel={channel.id} reply_chars={len(output)}")
                elif output:
                    _log("unprompted-skip", "generated message failed validation")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log("unprompted-skip", f"{type(exc).__name__}: {exc}")

    def reset_unprompted_task(channel=None) -> None:
        if channel is not None and getattr(channel, "guild", None) is not None:
            client.unprompted_channel = channel
        elif client.unprompted_channel is not None:
            channel = client.unprompted_channel
        elif DISCORD_ALLOWED_CHANNEL_IDS:
            channel = client.get_channel(DISCORD_ALLOWED_CHANNEL_IDS[0])
        if channel is None:
            _log("unprompted-skip", "no Discord channel is available yet")
            return
        client.unprompted_channel = channel
        if client.unprompted_task is not None:
            client.unprompted_task.cancel()
        client.unprompted_task = asyncio.create_task(unprompted_loop(channel))
        _log("unprompted", f"channel={channel.id} interval=4h")

    @client.event
    async def on_ready() -> None:
        _log("ready", f"logged in as {client.user}")
        _log("status", f"forwarding to {DISCORD_SERVER_URL}")
        if client.unprompted_task is None or client.unprompted_task.done():
            reset_unprompted_task()

    @client.event
    async def on_message(message) -> None:
        if client.user is None or not _mark_event(message):
            return
        bot_user_id = int(client.user.id)
        if not _should_handle(message, bot_user_id):
            return
        text = _message_text(message, bot_user_id)
        if not text:
            return
        reset_unprompted_task(message.channel)

        conversation_id = _conversation_id(message)
        profile_id = _profile_id(message)
        payload = _payload(message, text)
        try:
            async with message.channel.typing():
                result = await _post_chat(client.http_session, payload)
        except asyncio.CancelledError:
            await _cancel_remote(client.http_session, conversation_id, profile_id)
            raise

        error = compact_text(result.get("error"), 240)
        if error:
            _log("error", error)
            await _send_reply(message, error, allowed_mentions)
            return
        output = str(result.get("reply") or result.get("notice") or "").strip()
        if not output:
            return
        await _send_reply(message, output, allowed_mentions)

    client.run(DISCORD_BOT_TOKEN)
