"""Discord bot adapter for Akane.

This process stays lightweight and forwards messages to the running Akane
server over HTTP instead of importing the model/runtime directly.
"""

from __future__ import annotations

import asyncio
import json
from urllib import error as urlerror
from urllib import request as urlrequest

from app.config import (
    DISCORD_ALLOWED_CHANNEL_IDS,
    DISCORD_BOT_TOKEN,
    DISCORD_PREFIX,
    DISCORD_REPLY_TO_DMS,
    DISCORD_SERVER_URL,
)

_CHAT_TIMEOUT_SECONDS = 300
_INITIAL_SERVER_RETRY_DELAYS = (1.0, 2.0)


def _leading_mention_user_id(content: str) -> int | None:
    text = str(content or "").lstrip()
    if not text.startswith("<@"):
        return None
    end = text.find(">")
    if end <= 2:
        return None
    inner = text[2:end]
    if inner.startswith("!"):
        inner = inner[1:]
    return int(inner) if inner.isdigit() else None


def _log_terminal(label: str, text: str = "") -> None:
    message = f"[Akane][discord][{label}]"
    if text:
        message = f"{message} {text}"
    print(message, flush=True)


def _is_connection_refused_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return "connection refused" in lowered or "[errno 111]" in lowered


def _is_dm_message(message) -> bool:
    return getattr(message.guild, "id", None) is None


def _session_id_for_message(message) -> str:
    if _is_dm_message(message):
        return f"discord:dm:{message.author.id}"
    return f"discord:guild:{message.guild.id}:channel:{message.channel.id}"


def _normalize_message_content(message, bot_user_id: int, prefix: str) -> str:
    content = str(message.content or "").strip()
    if not content:
        return ""
    if prefix and content.lower().startswith(prefix.lower()):
        return content[len(prefix):].strip()
    mention_id = _leading_mention_user_id(content)
    if mention_id == bot_user_id:
        return content[content.find(">") + 1:].strip()
    return content


def _should_handle_message(message, bot_user_id: int, prefix: str) -> bool:
    if getattr(message.author, "bot", False):
        return False
    if _is_dm_message(message):
        return DISCORD_REPLY_TO_DMS
    if DISCORD_ALLOWED_CHANNEL_IDS and int(message.channel.id) not in DISCORD_ALLOWED_CHANNEL_IDS:
        return False
    content = str(message.content or "").strip()
    if not content:
        return False
    if prefix and content.lower().startswith(prefix.lower()):
        return True
    return _leading_mention_user_id(content) == bot_user_id


def _post_chat_request(prompt: str, session_id: str) -> dict:
    payload = json.dumps({"message": prompt, "skip_memory": True, "session_id": session_id}).encode("utf-8")
    req = urlrequest.Request(
        url=f"{DISCORD_SERVER_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=_CHAT_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        message = str(payload.get("detail") or payload.get("error") or body or exc).strip()
        return {"error": message or f"HTTP {exc.code}"}
    except urlerror.URLError as exc:
        return {"error": f"Could not reach Akane server at {DISCORD_SERVER_URL}: {exc.reason}"}
    except TimeoutError:
        return {"error": "Akane server timed out while generating a reply."}

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"error": "Akane server returned invalid JSON."}


def _stream_chat_request(prompt: str, session_id: str) -> dict:
    payload = json.dumps({"message": prompt, "skip_memory": True, "session_id": session_id}).encode("utf-8")
    req = urlrequest.Request(
        url=f"{DISCORD_SERVER_URL}/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=_CHAT_TIMEOUT_SECONDS) as resp:
            content_type = str(resp.headers.get("Content-Type", "") or "").lower()
            body = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        message = str(payload.get("detail") or payload.get("error") or body or exc).strip()
        return {"error": message or f"HTTP {exc.code}"}
    except urlerror.URLError as exc:
        return {"error": f"Could not reach Akane server at {DISCORD_SERVER_URL}: {exc.reason}"}
    except TimeoutError:
        return {"error": "Akane server timed out while generating a reply."}

    if "application/json" in content_type:
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": "Akane server returned invalid JSON."}

    final_reply = ""
    error_message = ""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", "") or "")
        if event_type == "error":
            error_message = str(event.get("error", "") or event.get("detail", "")).strip()
            break
        if event_type == "done":
            final_reply = str(event.get("reply", "") or "").strip()
            break
        if event_type == "delta" and not final_reply:
            final_reply = str(event.get("content", "") or "").strip()

    if error_message:
        return {"error": error_message}
    if final_reply:
        return {"reply": final_reply}
    return {"error": "Akane server returned no reply."}


def run_discord_bot() -> None:
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("Missing Discord bot token. Set AKANE_DISCORD_BOT_TOKEN or DISCORD_BOT_TOKEN in local secrets.")

    aiohttp = None
    try:
        import aiohttp as _aiohttp
        import discord
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        if exc.name != "aiohttp":
            raise SystemExit(
                "discord.py is not installed. Install it with `pip install discord.py` before running Discord mode."
            ) from exc
    else:
        aiohttp = _aiohttp

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    server_session = None
    http_timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120) if aiohttp is not None else None

    async def _post_chat_request_async(prompt: str, session_id: str) -> dict:
        nonlocal server_session
        retry_delays = (0.0, *_INITIAL_SERVER_RETRY_DELAYS)
        last_result = {"error": f"Could not reach Akane server at {DISCORD_SERVER_URL}."}
        for delay in retry_delays:
            if delay:
                await asyncio.sleep(delay)
            if aiohttp is None:
                result = await asyncio.to_thread(_stream_chat_request, prompt, session_id)
            else:
                if server_session is None or server_session.closed:
                    server_session = aiohttp.ClientSession(timeout=http_timeout)
                try:
                    async with server_session.post(
                        f"{DISCORD_SERVER_URL}/api/chat/stream",
                        json={"message": prompt, "skip_memory": True, "session_id": session_id},
                    ) as resp:
                        content_type = str(resp.headers.get("Content-Type", "") or "").lower()
                        if resp.status >= 400:
                            body = await resp.text()
                            try:
                                payload = json.loads(body)
                            except json.JSONDecodeError:
                                payload = {}
                            message = str(payload.get("detail") or payload.get("error") or body or resp.reason).strip()
                            result = {"error": message or f"HTTP {resp.status}"}
                        elif "application/json" in content_type:
                            body = await resp.text()
                            try:
                                payload = json.loads(body)
                            except json.JSONDecodeError:
                                payload = {}
                            result = payload if payload else {"error": "Akane server returned invalid JSON."}
                        else:
                            final_reply = ""
                            async for raw_line in resp.content:
                                line = raw_line.decode("utf-8", errors="ignore").strip()
                                if not line:
                                    continue
                                try:
                                    event = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                event_type = str(event.get("type", "") or "")
                                if event_type == "error":
                                    message = str(event.get("error", "") or event.get("detail", "")).strip()
                                    result = {"error": message or "Streaming failed."}
                                    break
                                if event_type == "done":
                                    final_reply = str(event.get("reply", "") or "").strip()
                                    break
                                if event_type == "delta" and not final_reply:
                                    final_reply = str(event.get("content", "") or "").strip()
                            else:
                                result = {"error": "Akane server returned no reply."}
                            if final_reply:
                                result = {"reply": final_reply}
                except Exception:
                    result = await asyncio.to_thread(_stream_chat_request, prompt, session_id)
            if not _is_connection_refused_error(result.get("error", "")):
                return result
            last_result = result
        return last_result

    @client.event
    async def on_ready() -> None:
        _log_terminal("ready", f"Logged in as {client.user}")
        _log_terminal("status", f"Forwarding messages to {DISCORD_SERVER_URL}")

    @client.event
    async def on_message(message) -> None:
        if client.user is None:
            return
        if not _should_handle_message(message, int(client.user.id), DISCORD_PREFIX):
            return

        prompt = _normalize_message_content(message, int(client.user.id), DISCORD_PREFIX)
        if not prompt:
            return

        scope = "dm" if _is_dm_message(message) else f"guild:{message.guild.id}/channel:{message.channel.id}"
        session_id = _session_id_for_message(message)
        _log_terminal("user", f"{message.author} ({scope}): {prompt}")
        _log_terminal("status", "Requesting reply from server...")

        async with message.channel.typing():
            result = await _post_chat_request_async(prompt, session_id)

        error = str(result.get("error", "") or result.get("detail", "")).strip()
        if error:
            _log_terminal("error", error)
            await message.reply(error, mention_author=False)
            return

        reply = str(result.get("reply", "") or "").strip()
        notice = str(result.get("notice", "") or "").strip()
        output = reply or notice
        if not output:
            return
        _log_terminal("assistant", output)
        await message.reply(output, mention_author=False)

    @client.event
    async def on_close() -> None:
        nonlocal server_session
        if server_session is not None and not server_session.closed:
            await server_session.close()

    client.run(DISCORD_BOT_TOKEN)
