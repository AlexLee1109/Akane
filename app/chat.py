"""Terminal chat interface for Akane."""

import os
import random
import select
import sys
import time
from collections import deque

from app.model_loader import LLM
from app.character import build_system_prompt
from app.memory import MEMORY_PATH, analyze_conversation_context, format_for_prompt, get_relationship_context, record_interaction, reload_from_disk
from app.generation import (
    HiddenTagStreamFilter,
    _generation_lock,
    build_runtime_context,
    capture_explicit_user_memories,
    finalize_model_response,
    generate_proactive,
    truncate_messages,
)
from app.config import CHAT_HISTORY_CONTEXT_TOKENS, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY, PROACTIVE_CHANCE, IDLE_INTERJECT_SECONDS

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"


def print_response_header():
    print(f"\n{BOLD}{CYAN}akane{RESET} > ", end="", flush=True)


def stream_and_print(stream) -> str:
    """Stream tokens, hide tags, print visible text. Returns full raw response."""
    full_response = []
    first_char = True
    stream_filter = HiddenTagStreamFilter()

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            token = delta["content"]
            full_response.append(token)
            visible = stream_filter.feed(token)
            if visible:
                if first_char:
                    visible = visible[0].upper() + visible[1:]
                    first_char = False
                print(visible, end="", flush=True)

    visible = stream_filter.flush()
    if visible:
        if first_char:
            visible = visible[0].upper() + visible[1:]
        print(visible, end="", flush=True)

    return "".join(full_response)


def _record_recent_interaction(messages: deque) -> None:
    """Update relationship/memory signals from the latest conversation window."""
    context = analyze_conversation_context(list(messages)[-20:])
    record_interaction(conversation_context=context)


def try_proactive(messages: list[dict], force: bool = False) -> bool:
    """Try to generate a proactive interjection. Returns True if one was made."""
    rel_ctx = get_relationship_context()
    hours_since = 0
    if rel_ctx["last_seen"]:
        from datetime import datetime
        last = datetime.fromisoformat(rel_ctx["last_seen"].replace("Z", "+00:00"))
        now = datetime.now(last.tzinfo)
        hours_since = (now - last).total_seconds() / 3600

    # Special check-in if user hasn't been seen in >24 hours and we have some history
    if hours_since > 24 and rel_ctx["interaction_count"] > 5 and random.random() < 0.3:
        checkin_messages = [
            f"Hey, hasn't heard from you in about {int(hours_since/24)} day{'s' if hours_since>=48 else ''}. Everything okay?",
            f"Long time no see! How are things?",
            f"It's been a while — just checking in to see how you're doing.",
        ]
        comment = random.choice(checkin_messages)
        print()
        print_response_header()
        print(comment)
        print()
        messages.append({"role": "assistant", "content": comment})
        return True

    if not force and random.random() > PROACTIVE_CHANCE:
        return False

    comment = generate_proactive(list(messages))
    if comment:
        print()
        print_response_header()
        print(comment)
        print()
        messages.append({"role": "assistant", "content": comment})
        return True
    return False


def wait_for_input(timeout: float) -> str | None:
    """Wait for user input with a timeout. Returns input string or None if timed out."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip()
    return None


def chat():
    print(f"\n{BOLD}{CYAN}Akane{RESET} — {DIM}your AI companion{RESET}")
    print(f"{DIM}Commands: '/memory' view memories, '/clear' reset context, '/reset' wipe all{RESET}\n")

    # Load model at startup
    print(f"{DIM}Loading model...{RESET}", end="", flush=True)
    from app.model_loader import ModelManager
    try:
        ModelManager.get_instance().ensure_loaded()
        print(f"\r{' ' * 40}\r", end="", flush=True)  # Clear loading message
    except Exception as e:
        return

    messages = deque(maxlen=20)  # Keep last 20 messages total
    last_interaction = time.time()
    next_idle_timeout = random.uniform(50, 120)  # randomized idle check

    while True:
        # Check if we should interject due to idle time
        if IDLE_INTERJECT_SECONDS > 0 and messages:
            elapsed = time.time() - last_interaction
            remaining = next_idle_timeout - elapsed
            if remaining > 0:
                user_input = wait_for_input(remaining)
                if user_input is None:
                    # Timed out — try to interject
                    if try_proactive(messages, force=True):
                        last_interaction = time.time()
                        next_idle_timeout = random.uniform(50, 120)
                        continue
                    else:
                        # Nothing to say, reset timer and wait for input
                        next_idle_timeout = random.uniform(50, 120)
                        last_interaction = time.time()
                        try:
                            user_input = input(f"{GREEN}you{RESET} > ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print(f"\n{DIM}bye!{RESET}")
                            break
                else:
                    # Got input during wait
                    if user_input == "":
                        continue
            else:
                # Already past idle threshold
                try_proactive(messages, force=True)
                last_interaction = time.time()
                next_idle_timeout = random.uniform(50, 120)
                try:
                    user_input = input(f"{GREEN}you{RESET} > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{DIM}bye!{RESET}")
                    break
        else:
            try:
                user_input = input(f"{GREEN}you{RESET} > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}bye!{RESET}")
                break

        if not user_input:
            continue

        last_interaction = time.time()

        if user_input.lower() in ("exit", "quit", "q"):
            print(f"{DIM}bye!{RESET}")
            break

        if user_input == "/memory":
            reload_from_disk()
            data = format_for_prompt()
            if data:
                print(f"\n{DIM}{data}{RESET}\n")
            else:
                print(f"\n{DIM}no memories yet{RESET}\n")
            continue

        if user_input == "/clear":
            messages.clear()
            print(f"\n{DIM}context cleared{RESET}\n")
            continue

        if user_input == "/reset":
            messages.clear()
            if MEMORY_PATH.exists():
                os.remove(MEMORY_PATH)
            print(f"\n{DIM}memory wiped and context cleared{RESET}\n")
            continue

        capture_explicit_user_memories(user_input)

        # Build system prompt with current memory
        system_prompt = build_system_prompt(build_runtime_context())

        chat_messages = truncate_messages(
            list(messages),  # pass the full deque as a list
            system_prompt,
            user_input,
            max_context_tokens=CHAT_HISTORY_CONTEXT_TOKENS
        )

        # Generate
        print_response_header()

        with _generation_lock:
            stream = LLM.create_chat_completion(
                messages=chat_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repeat_penalty=REPETITION_PENALTY,
                stream=True,
            )
            raw = stream_and_print(stream)

        print()  # newline

        cleaned = finalize_model_response(raw)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": cleaned})
        _record_recent_interaction(messages)


if __name__ == "__main__":
    chat()
