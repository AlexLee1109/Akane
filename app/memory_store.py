"""Persistent user memory system with in-memory caching and synchronous writes.

The store is intentionally conservative about extraction. Natural-language
understanding belongs to the model, which can emit hidden memory tags after it
has read the whole turn. This module validates, normalizes, deduplicates, and
persists those structured operations.
"""

import copy
import hashlib
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import orjson

MEMORY_PATH = Path(__file__).parent / "memory.json"

DEFAULT_MEMORY = {
    "schema_version": 2,
    "user": {},
    "preferences": [],
    "facts": [],
    "activities": {},
    "metadata": {
        "first_seen": None,
        "last_seen": None,
    }
}

_ENTRY_LIMITS = {
    "preferences": 24,
    "facts": 32,
}
_VALID_MEMORY_CATEGORIES = {"user", "preferences", "facts"}
_TAG_PAIRS = {
    "MEM": "[/MEM]",
    "FORGET": "[/FORGET]",
    "PROJECT": "[/PROJECT]",
}
_CONTROL_CHARS = str.maketrans({chr(i): " " for i in range(32) if chr(i) not in "\n\t"})
_LOW_VALUE_PUNCT = str.maketrans({ch: " " for ch in "\"'`.,:;!?()[]{}<>|/\\@#$%^&*=+~"})
_DISCORD_CONTEXT_PREFIX = "discord context."
_TRANSIENT_MEMORY_PHRASES = (
    "asked me",
    "asking me",
    "asked about",
    "current message",
    "current request",
    "right now",
    "this chat",
    "this conversation",
    "this message",
    "this request",
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "needs help",
    "wants help",
    "wants me to",
    "speaker id",
    "discord context",
)
_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z' -]{0,47})(?:[.!?,]|$)", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-z][A-Za-z' -]{0,47})(?:[.!?,]|$)", re.IGNORECASE),
)
_FAVORITE_RE = re.compile(
    r"\bmy favorite\s+([A-Za-z][A-Za-z0-9 _/-]{1,40})\s+is\s+([^.!?\n]{2,120})",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:i|we)\s+(?:really\s+|especially\s+)?(like|love|enjoy|prefer)\s+([^.!?\n]{2,120})",
    re.IGNORECASE,
)
_USE_RE = re.compile(
    r"\b(?:i|we)\s+(?:use|run|am running|have)\s+([^.!?\n]{2,120})",
    re.IGNORECASE,
)
_STABLE_USE_HINTS = {
    "pc", "computer", "mac", "windows", "linux", "raspberry", "pi", "gpu", "cpu",
    "ram", "model", "qwen", "llama", "python", "discord", "vscode", "server",
    "laptop", "desktop", "keyboard", "monitor",
}
_VAGUE_MEMORY_VALUES = {
    "it", "that", "this", "those", "them", "you", "your", "me", "him", "her",
    "stuff", "things", "something", "anything", "everything",
}


def _fresh_default_memory() -> dict:
    return copy.deepcopy(DEFAULT_MEMORY)


def _ensure_memory_shape(data: dict) -> dict:
    """Normalize loaded memory data to the current schema."""
    normalized = _fresh_default_memory()
    if not isinstance(data, dict):
        return normalized

    if isinstance(data.get("schema_version"), int):
        normalized["schema_version"] = max(2, data["schema_version"])

    for key in ("user", "activities"):
        if isinstance(data.get(key), dict):
            normalized[key].update(copy.deepcopy(data[key]))

    if isinstance(data.get("metadata"), dict):
        for key in normalized["metadata"]:
            normalized["metadata"][key] = copy.deepcopy(data["metadata"].get(key))

    for key in ("preferences", "facts"):
        if isinstance(data.get(key), list):
            normalized[key] = copy.deepcopy(data[key])

    return normalized


def _touch_metadata(data: dict) -> None:
    metadata = data.setdefault("metadata", {})
    now = _now()
    if metadata.get("first_seen") is None:
        metadata["first_seen"] = now
    metadata["last_seen"] = now


def _new_list_entry(content: str) -> dict:
    now = _now()
    return {
        "id": _entry_id(content),
        "content": content,
        "created": now,
        "last_seen": now,
        "mentions": 1,
    }


def _touch_list_entry(entries: list[dict], content: str) -> bool:
    fingerprint = _entry_fingerprint(content)
    for entry in entries:
        existing = _normalize_entry_text(entry.get("content", ""))
        if _entries_match(existing, content, fingerprint):
            entry["mentions"] = entry.get("mentions", 1) + 1
            entry["last_seen"] = _now()
            entry["id"] = entry.get("id") or _entry_id(content)
            if _is_better_entry_text(content, existing):
                entry["content"] = content
            return True
    return False


def _normalize_person_name(value: str) -> str:
    value = " ".join(value.strip().split()).lower()
    if not value:
        return ""
    out: list[str] = []
    cap_next = True
    for ch in value:
        if cap_next and "a" <= ch <= "z":
            out.append(ch.upper())
            cap_next = False
            continue
        out.append(ch)
        if ch in {" ", "-", "'"}:
            cap_next = True
        elif ch.isalpha():
            cap_next = False
    return "".join(out)


def _normalize_user_field(key: str, value: str) -> str:
    normalized_key = key.strip().lower()
    normalized_value = value.strip()
    if normalized_key == "name":
        return _normalize_person_name(normalized_value)
    return normalized_value


def _normalize_entry_text(value: str) -> str:
    text = str(value or "").translate(_CONTROL_CHARS).strip()
    return " ".join(text.split())


def _entry_fingerprint(value: str) -> str:
    cleaned = _normalize_entry_text(value).lower().translate(_LOW_VALUE_PUNCT)
    cleaned = re.sub(r"^(?:the\s+)?(?:user|speaker)\s+", "", cleaned)
    return " ".join(cleaned.split())


def _entry_id(value: str) -> str:
    digest = hashlib.sha256(_entry_fingerprint(value).encode("utf-8")).hexdigest()
    return digest[:16]


def _entries_match(existing: str, candidate: str, candidate_fingerprint: str | None = None) -> bool:
    existing_norm = _entry_fingerprint(existing)
    candidate_norm = candidate_fingerprint if candidate_fingerprint is not None else _entry_fingerprint(candidate)
    if not existing_norm or not candidate_norm:
        return False
    return existing_norm == candidate_norm


def _is_better_entry_text(candidate: str, existing: str) -> bool:
    candidate = _normalize_entry_text(candidate)
    existing = _normalize_entry_text(existing)
    if not existing:
        return True
    if candidate == existing:
        return False
    return len(candidate) > len(existing) and len(candidate) <= 220


def _memory_entry_allowed(category: str, content: str) -> bool:
    normalized = _normalize_entry_text(content)
    if category not in {"preferences", "facts"}:
        return False
    if not normalized or len(normalized) < 4 or len(normalized) > 220:
        return False
    if any(marker in normalized for marker in ("[MEM]", "[/MEM]", "[PROJECT]", "[/PROJECT]", "[FORGET]", "[/FORGET]")):
        return False
    if "<" in normalized or ">" in normalized:
        return False
    lowered = normalized.lower()
    if any(phrase in lowered for phrase in _TRANSIENT_MEMORY_PHRASES):
        return False
    if lowered.startswith(("akane ", "assistant ", "the assistant ")):
        return False
    if "?" in normalized:
        return False
    words = normalized.translate(_LOW_VALUE_PUNCT).split()
    if len(words) < 2:
        return False
    if len(set(word.lower() for word in words)) <= 1 and len(words) > 2:
        return False
    return True


def _project_detail_allowed(content: str) -> bool:
    normalized = _normalize_entry_text(content)
    if not normalized or len(normalized) > 180:
        return False
    if "?" in normalized or "[" in normalized or "]" in normalized or "<" in normalized or ">" in normalized:
        return False
    return True


def _clean_extracted_value(value: str) -> str:
    return _normalize_entry_text(value).strip(" ,;:-")


def _clean_natural_memory_value(value: str) -> str:
    value = _clean_extracted_value(value)
    value = re.split(r"\s+(?:and|but|because)\s+i\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\s+(?:too|as well)$", "", value, flags=re.IGNORECASE).strip()
    value = value.strip("\"'` ")
    return value[:120].strip()


def _specific_enough_memory_value(value: str) -> bool:
    normalized = _entry_fingerprint(value)
    if not normalized or normalized in _VAGUE_MEMORY_VALUES:
        return False
    words = normalized.split()
    return len(words) >= 2 or len(normalized) >= 4


def _looks_like_valid_name(value: str) -> bool:
    candidate = _clean_extracted_value(value)
    parts = candidate.replace("-", " ").replace("'", " ").split()
    return bool(candidate) and len(candidate) <= 48 and not any(ch.isdigit() for ch in candidate) and 1 <= len(parts) <= 4 and all(part.isalpha() for part in parts)


def _normalize_user_key(value: str) -> str:
    key = _clean_extracted_value(value).lower()
    key = key.removeprefix("my ").strip()
    key = key.replace("'", "")
    key = "_".join(part for part in key.split() if part)
    return key[:48]


def _add_unique_op(ops: list[tuple[str, str]], seen: set[tuple[str, str]], op: tuple[str, str]) -> None:
    if op not in seen:
        seen.add(op)
        ops.append(op)


def _extract_tag_contents(text: str, opener: str, closer: str) -> list[str]:
    source = str(text or "")
    if not source:
        return []
    haystack = source.lower()
    open_tag = opener.lower()
    close_tag = closer.lower()
    if open_tag not in haystack:
        return []
    parts: list[str] = []
    start = 0
    while True:
        open_idx = haystack.find(open_tag, start)
        if open_idx == -1:
            return parts
        content_idx = open_idx + len(opener)
        close_idx = haystack.find(close_tag, content_idx)
        if close_idx == -1:
            return parts
        content = source[content_idx:close_idx].strip()
        if content:
            parts.append(content)
        start = close_idx + len(closer)


def _decode_json_object(value: str) -> dict | None:
    raw = str(value or "").strip()
    if not raw or raw[0] != "{" or raw[-1] != "}":
        return None
    try:
        decoded = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _parse_memory_payload(raw: str) -> tuple[str, str] | None:
    """Parse one structured [MEM] body into a store operation."""
    text = str(raw or "").strip()
    if not text:
        return None

    decoded = _decode_json_object(text)
    if decoded is not None:
        category = str(decoded.get("category") or decoded.get("type") or "").strip().lower()
        content = decoded.get("content")
        if category in {"preference", "preferences"}:
            category = "preferences"
        elif category in {"fact", "facts"}:
            category = "facts"
        elif category in {"user", "profile"}:
            category = "user"
        if category == "user":
            key = str(decoded.get("key") or "").strip()
            value = str(decoded.get("value") if decoded.get("value") is not None else content or "").strip()
            if key and value:
                return "user", f"{key}: {value}"
            return None
        if category in {"preferences", "facts"} and content:
            return category, str(content).strip()
        return None

    if ":" not in text:
        return None
    category, content = text.split(":", 1)
    category = category.strip().lower()
    content = content.strip()
    if category in {"preference", "preferences"}:
        return "preferences", content
    if category in {"fact", "facts"}:
        return "facts", content
    if category == "user":
        return "user", content
    if category == "name":
        return "user", f"name: {content}"
    return None


def _looks_like_discord_wrapped_prompt(text: str) -> bool:
    return str(text or "").lstrip().lower().startswith(_DISCORD_CONTEXT_PREFIX)


def _natural_language_memory_ops(user_text: str) -> list[tuple[str, str]]:
    """Extract only high-confidence first-person memory from raw user text."""
    text = _normalize_entry_text(user_text)
    if not text or _looks_like_discord_wrapped_prompt(text):
        return []

    ops: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = _clean_natural_memory_value(match.group(1))
        if _looks_like_valid_name(name):
            _add_unique_op(ops, seen, ("user", f"name: {name}"))
            break

    favorite = _FAVORITE_RE.search(text)
    if favorite:
        subject = _clean_natural_memory_value(favorite.group(1)).lower()
        value = _clean_natural_memory_value(favorite.group(2))
        if subject and value:
            _add_unique_op(ops, seen, ("preferences", f"Favorite {subject}: {value}"))

    for verb, raw_value in _PREFERENCE_RE.findall(text):
        value = _clean_natural_memory_value(raw_value)
        lowered_value = value.lower()
        if (
            not value
            or not _specific_enough_memory_value(value)
            or any(phrase in lowered_value for phrase in _TRANSIENT_MEMORY_PHRASES)
        ):
            continue
        label = "Prefers" if verb.lower() == "prefer" else "Likes"
        _add_unique_op(ops, seen, ("preferences", f"{label} {value}"))

    for raw_value in _USE_RE.findall(text):
        value = _clean_natural_memory_value(raw_value)
        tokens = set(value.lower().translate(_LOW_VALUE_PUNCT).split())
        if value and _specific_enough_memory_value(value) and tokens & _STABLE_USE_HINTS:
            _add_unique_op(ops, seen, ("facts", f"Uses {value}"))

    return ops


def _extract_memory_ops_from_user_text(
    user_text: str,
    *,
    include_natural: bool = False,
) -> tuple[list[tuple[str, str]], list[str], list[tuple[str, str]]]:
    """Extract only explicit structured memory tags from raw user text.

    Natural-language messages are no longer scanned for words or prefixes. The
    assistant's hidden tags, emitted after semantic interpretation, are the
    normal memory extraction path. This parser exists for manual/API callers
    that send structured tags directly.
    """
    text = str(user_text or "").strip()
    if not text:
        return [], [], []

    mem_ops: list[tuple[str, str]] = []
    forget_queries: list[str] = []
    project_ops: list[tuple[str, str]] = []
    seen_mem: set[tuple[str, str]] = set()
    seen_forget: set[str] = set()
    seen_projects: set[tuple[str, str]] = set()

    for raw in _extract_tag_contents(text, "[MEM]", _TAG_PAIRS["MEM"]):
        parsed = _parse_memory_payload(raw)
        if parsed:
            _add_unique_op(mem_ops, seen_mem, parsed)

    if include_natural:
        for parsed in _natural_language_memory_ops(text):
            _add_unique_op(mem_ops, seen_mem, parsed)

    for raw in _extract_tag_contents(text, "[FORGET]", _TAG_PAIRS["FORGET"]):
        query = _clean_extracted_value(raw).lower()
        if query and query not in seen_forget:
            seen_forget.add(query)
            forget_queries.append(query)

    for raw in _extract_tag_contents(text, "[PROJECT]", _TAG_PAIRS["PROJECT"]):
        if ":" in raw:
            name, detail = raw.split(":", 1)
        else:
            name, detail = raw, ""
        name = _clean_extracted_value(name).lower()
        detail = _clean_extracted_value(detail).lower()
        if name:
            _add_unique_op(project_ops, seen_projects, (name, detail))

    return mem_ops, forget_queries, project_ops


def _enforce_entry_limit(data: dict, category: str) -> None:
    limit = _ENTRY_LIMITS.get(category)
    if not limit or category not in data or len(data[category]) <= limit:
        return
    entries = data[category]
    ranked = sorted(
        entries,
        key=lambda entry: (
            -int(entry.get("mentions", 1)),
            str(entry.get("last_seen", entry.get("created", ""))),
            str(entry.get("created", "")),
        ),
        reverse=True,
    )
    keep_ids = {id(entry) for entry in ranked[:limit]}
    data[category] = [entry for entry in entries if id(entry) in keep_ids]


# ---------------------------------------------------------------------------
# MemoryStore — thread-safe singleton
# ---------------------------------------------------------------------------

class MemoryStore:
    """Thread-safe memory store with prompt caching."""

    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self.data = None
        self.dirty = False
        self._flush_lock = threading.RLock()
        self._prompt_cache = None
        self._prompt_cache_gen = -1
        self._load_initial()

    def _load_initial(self):
        """Load memory from disk once at startup."""
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        """Load memory from disk into the in-memory store."""
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.data = _ensure_memory_shape(orjson.loads(f.read()))
            except Exception as e:
                print(f"Error loading memory: {e}, starting fresh")
                self.data = _fresh_default_memory()
                self._flush(force=True)
        else:
            self.data = _fresh_default_memory()
            self._flush(force=True)

    def reload(self) -> None:
        """Refresh the in-memory store from disk and invalidate prompt cache."""
        with self._flush_lock:
            if self.dirty:
                self._flush(force=True)
            self._reload_from_disk()
            self._prompt_cache = None
            self._prompt_cache_gen += 1

    def get(self) -> dict:
        """Get a deep copy of current memory state."""
        with self._flush_lock:
            return copy.deepcopy(self.data)

    def update(self, updater_func: Callable[[dict], None], *, invalidate_prompt_cache: bool = True) -> None:
        """Atomically update memory in-place and mark as dirty."""
        with self._flush_lock:
            result = updater_func(self.data)
            self.dirty = True
            if invalidate_prompt_cache:
                self._prompt_cache_gen += 1
            return result

    def _flush(self, force: bool = False) -> bool:
        """Write memory to disk if dirty or forced. Returns True if flushed."""
        with self._flush_lock:
            if not self.dirty and not force:
                return False
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
                temp_path.replace(self.path)
                self.dirty = False
                return True
            except Exception as e:
                print(f"Error flushing memory: {e}")
                return False

    def shutdown(self):
        """Flush remaining dirty data."""
        self._flush(force=True)


_store = None
_store_lock = threading.Lock()


def get_store() -> MemoryStore:
    """Get the global memory store instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryStore()
    return _store


def reload_from_disk() -> None:
    """Refresh the singleton memory store from the current on-disk JSON file."""
    get_store().reload()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_tag_operations(
    *,
    mem_ops: list[tuple[str, str]],
    forget_queries: list[str],
    project_ops: list[tuple[str, str]],
) -> bool:
    """Apply parsed memory tag operations in one authoritative place."""
    if not (mem_ops or forget_queries or project_ops):
        return False

    store = get_store()

    def update(data):
        _touch_metadata(data)

        for mem_cat, content in mem_ops:
            mem_cat = str(mem_cat or "").strip().lower()
            if mem_cat in {"preference", "preferences"}:
                mem_cat = "preferences"
            elif mem_cat in {"fact", "facts"}:
                mem_cat = "facts"
            elif mem_cat not in _VALID_MEMORY_CATEGORIES:
                continue

            content = _normalize_entry_text(content)
            if mem_cat == "user":
                if ":" in content:
                    key, value = content.split(":", 1)
                    key = _normalize_user_key(key)
                    if not key:
                        continue
                    norm_value = _normalize_user_field(key, value)
                    if not norm_value or len(norm_value) > 160:
                        continue
                    if key == "name" and not _looks_like_valid_name(norm_value):
                        continue
                    data["user"][key] = norm_value
                continue
            if not _memory_entry_allowed(mem_cat, content):
                continue
            if not _touch_list_entry(data[mem_cat], content):
                data[mem_cat].append(_new_list_entry(content))
            _enforce_entry_limit(data, mem_cat)

        for query in forget_queries:
            query_lower = query.lower()
            for category in ("preferences", "facts"):
                data[category] = [
                    entry for entry in data[category]
                    if query_lower not in entry.get("content", "").lower()
                ]
            keys_to_remove = [
                key for key, value in data["user"].items()
                if query_lower in key.lower() or query_lower in str(value).lower()
            ]
            for key in keys_to_remove:
                del data["user"][key]
            activity_names = [
                name for name, activity in data["activities"].items()
                if query_lower in name or any(query_lower in d.lower() for d in activity.get("details", []))
            ]
            for name in activity_names:
                del data["activities"][name]

        for name, detail in project_ops:
            name = _normalize_entry_text(name).lower()
            detail = _normalize_entry_text(detail).lower()
            activity = data["activities"].get(name)

            if detail in {"done", "inactive", "active", "resume", "resumed", "delete", "remove"}:
                if detail in {"active", "resume", "resumed"} and activity is None:
                    activity = data["activities"][name] = {"details": [], "status": "active", "created": _now()}
                if activity is None:
                    continue
                if detail == "done":
                    activity["status"] = "done"
                    activity["completed"] = _now()
                elif detail == "inactive":
                    activity["status"] = "inactive"
                    activity["updated"] = _now()
                elif detail in {"active", "resume", "resumed"}:
                    activity["status"] = "active"
                    activity["updated"] = _now()
                elif detail in {"delete", "remove"}:
                    del data["activities"][name]
                continue

            if activity is None:
                activity = data["activities"][name] = {"details": [], "status": "active", "created": _now()}
            if detail and detail not in activity["details"] and _project_detail_allowed(detail):
                activity["details"].append(detail)
                activity["updated"] = _now()

    store.update(update)
    store._flush(force=True)
    return True


def remember_user_message(user_text: str, *, allow_natural: bool = True) -> bool:
    """Apply explicit structured memory tags from a raw user message."""
    mem_ops, forget_queries, project_ops = _extract_memory_ops_from_user_text(
        user_text,
        include_natural=allow_natural,
    )
    if not mem_ops and not forget_queries and not project_ops:
        return False
    return apply_tag_operations(
        mem_ops=mem_ops,
        forget_queries=forget_queries,
        project_ops=project_ops,
    )


def record_interaction() -> None:
    """Compatibility shim for older server imports; does not count interactions."""
    store = get_store()

    def update(data):
        _touch_metadata(data)

    store.update(update)
    store._flush(force=True)


def get_all() -> dict:
    """Get all memory data."""
    return get_store().get()


def format_for_prompt() -> str:
    """Format all memories as a string for the system prompt."""
    store = get_store()
    with store._flush_lock:
        data = store.data
        cache_gen = store._prompt_cache_gen
        prompt_cache = store._prompt_cache

    if prompt_cache is not None and store._prompt_cache_gen == cache_gen:
        return prompt_cache

    sections = []
    user = data.get("user", {})

    def _append_unique(lines: list[str], seen: set[str], label: str, value: str) -> None:
        cleaned = " ".join(str(value or "").split()).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        lines.append(f"  - {label}: {cleaned}")

    # --- User Preferences ---
    preference_lines: list[str] = []
    seen_preferences: set[str] = set()

    name = str(user.get("name", "")).strip()
    if name:
        _append_unique(preference_lines, seen_preferences, "Name", name)

    for key, value in user.items():
        if key == "name":
            continue
        lowered_key = str(key).lower()
        if any(token in lowered_key for token in ("communication", "tone", "style")):
            _append_unique(preference_lines, seen_preferences, "Preferred communication style", value)
        elif any(token in lowered_key for token in ("game", "genre")):
            _append_unique(preference_lines, seen_preferences, "Favorite games/genres", value)
        elif any(token in lowered_key for token in ("task", "workflow", "focus")):
            _append_unique(preference_lines, seen_preferences, "Common tasks", value)
        else:
            _append_unique(preference_lines, seen_preferences, str(key).replace("_", " ").title(), value)

    game_prefs = [
        entry.get("content", "")
        for entry in data.get("preferences", [])
        if any(token in entry.get("content", "").lower() for token in ("game", "genre"))
    ]
    if game_prefs:
        _append_unique(preference_lines, seen_preferences, "Favorite games/genres", "; ".join(game_prefs[:3]))

    common_tasks = list(data.get("activities", {}).keys())[:4]
    if common_tasks:
        _append_unique(preference_lines, seen_preferences, "Common tasks", "; ".join(common_tasks))

    for entry in data.get("preferences", [])[:6]:
        content = str(entry.get("content", "")).strip()
        if any(token in content.lower() for token in ("game", "genre")):
            continue
        _append_unique(preference_lines, seen_preferences, "Preference", content)

    if preference_lines:
        sections.append("User Preferences:\n" + "\n".join(preference_lines))

    # --- System Context ---
    active = {k: v for k, v in data.get("activities", {}).items() if v.get("status") == "active"}
    if active:
        context_lines = []
        for activity_name, proj in active.items():
            details = "; ".join(proj["details"][:2]).strip()
            if details:
                context_lines.append(f"  - Recent files or projects: {activity_name} ({details})")
            else:
                context_lines.append(f"  - Recent files or projects: {activity_name}")
        sections.append("System Context:\n" + "\n".join(context_lines))

    # --- Persistent Facts ---
    fact_lines: list[str] = []
    seen_facts: set[str] = set()
    for key, value in user.items():
        lowered_key = str(key).lower()
        if any(token in lowered_key for token in ("hardware", "spec", "schedule", "habit", "routine")):
            _append_unique(fact_lines, seen_facts, str(key).replace("_", " ").title(), value)
    for entry in data.get("facts", [])[:8]:
        _append_unique(fact_lines, seen_facts, "Fact", entry.get("content", ""))
    if fact_lines:
        sections.append("Persistent Facts:\n" + "\n".join(fact_lines))

    result = "\n\n".join(sections) if sections else ""

    with store._flush_lock:
        store._prompt_cache = result
        store._prompt_cache_gen = cache_gen

    return result


def shutdown() -> None:
    """Shutdown the memory store, flush any pending writes."""
    global _store
    if _store:
        _store.shutdown()
