"""Diagnostic startup by default; explicit `chat` starts local inference."""

import argparse
from pathlib import Path
from uuid import UUID

from .conversation.model import INITIAL_MODEL_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Akane Next text interaction model")
    parser.add_argument("command", nargs="?", choices=["chat", "sessions", "delete-session"])
    parser.add_argument("--model", type=Path, help="Existing Qwen3.5-9B-Q4_K_M GGUF")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, help="Private directory for JSON session files")
    parser.add_argument("--profile", type=UUID, help="Explicit stable local profile UUID")
    parser.add_argument("--session", type=UUID, help="Existing session UUID to resume or delete")
    parser.add_argument("--auto-delegation", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable model-driven informational delegation")
    parser.add_argument('--decision-protocol',choices=('union','mode-first','intent-only'),default='union',
                        help='Experimental control format; automatic delegation remains opt-in')
    arguments = parser.parse_args(argv)
    if (arguments.data_dir is None) != (arguments.profile is None):
        parser.error("--data-dir and --profile must be supplied together")
    if arguments.session is not None and arguments.data_dir is None:
        parser.error("--session requires --data-dir and --profile")
    if arguments.command in ("sessions", "delete-session"):
        if arguments.data_dir is None:
            parser.error("session commands require --data-dir and --profile")
        if arguments.command == "delete-session" and arguments.session is None:
            parser.error("delete-session requires --session")
        from .continuity.json_store import JSONContinuityStore
        try:
            with JSONContinuityStore(arguments.data_dir, arguments.profile) as store:
                if arguments.command == "delete-session":
                    store.delete_session(arguments.session)
                    print(f"Deleted session {arguments.session}.")
                else:
                    for session in store.list_sessions():
                        print(f"{session.conversation_id}  turns={session.revision}  created={session.created_at.isoformat()}")
            return 0
        except (OSError, ValueError, RuntimeError, LookupError) as error:
            parser.exit(1, f"Akane Next: {error}\n")
    if arguments.command == "chat":
        if arguments.model is None:
            parser.error("chat requires --model pointing to the existing local GGUF")
        if not 0 < arguments.max_tokens < arguments.context_size:
            parser.error("require 0 < --max-tokens < --context-size")
        from .terminal import chat
        try:
            return chat(
                arguments.model, context_size=arguments.context_size,
                max_tokens=arguments.max_tokens,
                data_dir=arguments.data_dir, profile_id=arguments.profile,
                conversation_id=arguments.session, automatic_delegation=arguments.auto_delegation,
                decision_protocol=arguments.decision_protocol,
            )
        except (ImportError, OSError, ValueError, RuntimeError, LookupError) as error:
            parser.exit(1, f"Akane Next: {error}\n")
    print("Akane Next foundation ready (text only; no model loaded).")
    print(f"Model target: {INITIAL_MODEL_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

