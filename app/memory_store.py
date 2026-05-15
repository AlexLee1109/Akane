"""Persistent user memory system with in-memory caching and synchronous writes."""

import copy
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import orjson

MEMORY_PATH = Path(__file__).parent / "memory.json"

DEFAULT_MEMORY = {
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

_GENERIC_MEMORY_PREFIXES = (
    "what ", "why ", "how ", "when ", "where ", "who ", "should ", "could ", "would ",
    "can you ", "do you ", "is it ", "are you ",
)
_GENERIC_MEMORY_PHRASES = (
    "what do you think", "tell me more", "show me", "go deeper", "full breakdown",
    "can you help", "i need help", "look at", "check the code", "read the file",
    "open vscode", "open the project", "thermal throttling", "reply faster",
)
_CLAUSE_SPLIT_CHARS = ".!?\n"
_TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "i", "if",
    "in", "into", "is", "it", "me", "my", "of", "on", "or", "our", "so", "that", "the", "this",
    "to", "was", "we", "with", "you", "your",
}
_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in "\"'`.,:;!?()[]{}<>|/\\@#$%^&*=+~"})
_NAME_PREFIXES = ("my name is ", "name's ", "name is ", "call me ", "i am ", "i'm ", "i go by ")
_PREFERENCE_PREFIXES = (
    ("i prefer ", "prefers "),
    ("i like ", "likes "),
    ("i love ", "loves "),
    ("i enjoy ", "enjoys "),
    ("i want ", "wants "),
    ("i need ", "needs "),
    ("i don't like ", "doesn't like "),
    ("i do not like ", "doesn't like "),
    ("i hate ", "doesn't like "),
)
_FACT_PREFIXES = (
    ("i use ", "uses "),
    ("i'm using ", "uses "),
    ("im using ", "uses "),
    ("i have ", "has "),
    ("i'm on ", "uses "),
    ("im on ", "uses "),
    ("i run ", "runs "),
)
_PROJECT_PREFIXES = (
    "i am working on ", "i'm working on ", "im working on ",
    "i am building ", "i'm building ", "im building ",
    "i am making ", "i'm making ", "im making ",
    "i am creating ", "i'm creating ", "im creating ",
)
_MEMORY_WRAPPERS = (
    "please remember that ",
    "remember that ",
    "remember, ",
    "remember ",
    "you should remember that ",
    "keep in mind that ",
    "keep in mind ",
    "note that ",
)


def _fresh_default_memory() -> dict:
    return copy.deepcopy(DEFAULT_MEMORY)


def _ensure_memory_shape(data: dict) -> dict:
    """Normalize loaded memory data to the current schema."""
    normalized = _fresh_default_memory()
    if not isinstance(data, dict):
        return normalized

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
    return {"content": content, "created": _now(), "mentions": 1}


def _touch_list_entry(entries: list[dict], content: str) -> bool:
    normalized = _normalize_entry_text(content)
    normalized_tokens = _entry_token_set(normalized)
    for entry in entries:
        existing = _normalize_entry_text(entry.get("content", ""))
        if _entries_match(existing, normalized, normalized_tokens):
            entry["mentions"] = entry.get("mentions", 1) + 1
            entry["last_seen"] = _now()
            if len(normalized) > len(existing):
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
    return " ".join(str(value or "").strip().split())


def _entry_token_set(value: str) -> set[str]:
    return {
        token.lower()
        for token in _tokenize_words(_normalize_entry_text(value))
        if len(token) > 2 and token.lower() not in _TOKEN_STOPWORDS
    }


def _entries_match(existing: str, candidate: str, candidate_tokens: set[str] | None = None) -> bool:
    existing_norm = _normalize_entry_text(existing).lower()
    candidate_norm = _normalize_entry_text(candidate).lower()
    if not existing_norm or not candidate_norm:
        return False
    if existing_norm == candidate_norm:
        return True
    existing_tokens = _entry_token_set(existing_norm)
    candidate_tokens = candidate_tokens if candidate_tokens is not None else _entry_token_set(candidate_norm)
    if not existing_tokens or not candidate_tokens:
        return False
    overlap = len(existing_tokens & candidate_tokens)
    smaller = min(len(existing_tokens), len(candidate_tokens))
    return smaller > 0 and overlap >= smaller and abs(len(existing_tokens) - len(candidate_tokens)) <= 1


def _looks_like_generic_memory(content: str) -> bool:
    lowered = _normalize_entry_text(content).lower()
    if not lowered:
        return True
    if "?" in lowered or "[" in lowered or "]" in lowered or "<" in lowered or ">" in lowered:
        return True
    if any(lowered.startswith(prefix) for prefix in _GENERIC_MEMORY_PREFIXES):
        return True
    return any(phrase in lowered for phrase in _GENERIC_MEMORY_PHRASES)


def _memory_entry_allowed(category: str, content: str) -> bool:
    normalized = _normalize_entry_text(content)
    lowered = normalized.lower()
    if not normalized or len(normalized) < 4 or len(normalized) > 180:
        return False
    if _looks_like_generic_memory(normalized):
        return False

    tokens = _entry_token_set(normalized)
    if category == "preferences":
        return bool(tokens) and any(
            lowered.startswith(prefix)
            for prefix in ("prefers ", "likes ", "loves ", "enjoys ", "wants ", "needs ", "doesn't like ", "does not like ")
        )
    if category == "facts":
        starts_like_fact = lowered.startswith(("uses ", "has ", "runs ", "works on ", "working on "))
        return (
            (starts_like_fact and len(tokens) >= 2)
            or (len(tokens) >= 3 and any(char.isdigit() for char in normalized))
        )
    return True


def _clean_extracted_value(value: str) -> str:
    return _normalize_entry_text(value).strip(" ,;:-")


def _looks_like_valid_name(value: str) -> bool:
    candidate = _clean_extracted_value(value)
    parts = candidate.replace("-", " ").replace("'", " ").split()
    return bool(candidate) and len(candidate) <= 48 and not any(ch.isdigit() for ch in candidate) and 1 <= len(parts) <= 4 and all(part.isalpha() for part in parts)


def _split_user_clauses(text: str) -> list[str]:
    source = str(text or "")
    for char in _CLAUSE_SPLIT_CHARS:
        source = source.replace(char, "\n")
    return [clause for part in source.splitlines() if (clause := _clean_extracted_value(part))]


def _tokenize_words(text: str) -> list[str]:
    return " ".join(str(text or "").translate(_PUNCT_TRANSLATION).split()).split()


def _memory_clause_variants(clause: str) -> list[str]:
    """Return a small set of durable-memory phrasings worth parsing."""
    cleaned = _clean_extracted_value(clause)
    if not cleaned:
        return []
    lowered = cleaned.lower()
    for wrapper in _MEMORY_WRAPPERS:
        if lowered.startswith(wrapper):
            unwrapped = _clean_extracted_value(cleaned[len(wrapper):])
            return [cleaned, unwrapped] if unwrapped and unwrapped != cleaned else [cleaned]
    return [cleaned]


def _normalize_user_key(value: str) -> str:
    key = _clean_extracted_value(value).lower()
    key = key.removeprefix("my ").strip()
    key = key.replace("'", "")
    key = "_".join(part for part in key.split() if part)
    return key[:48]


def _extract_user_field_from_clause(clause: str) -> tuple[str, str] | None:
    lowered = clause.lower()
    for connector in (" is ", " are "):
        if connector not in lowered:
            continue
        left, right = clause.split(connector, 1)
        left_lower = left.strip().lower()
        if not left_lower.startswith("my "):
            continue
        value = _clean_extracted_value(right)
        key = _normalize_user_key(left)
        if not key or not value or len(value) > 120:
            return None
        if key == "name" and not _looks_like_valid_name(value):
            return None
        return key, value
    return None


def _add_unique_op(ops: list[tuple[str, str]], seen: set[tuple[str, str]], op: tuple[str, str]) -> None:
    if op not in seen:
        seen.add(op)
        ops.append(op)


def _extract_memory_ops_from_user_text(user_text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    text = str(user_text or "").strip()
    if not text:
        return [], []

    mem_ops: list[tuple[str, str]] = []
    project_ops: list[tuple[str, str]] = []
    seen_mem: set[tuple[str, str]] = set()
    seen_projects: set[tuple[str, str]] = set()

    for raw_clause in _split_user_clauses(text):
        for clause in _memory_clause_variants(raw_clause):
            lowered = clause.lower()

            for prefix in _NAME_PREFIXES:
                if not lowered.startswith(prefix):
                    continue
                name = _clean_extracted_value(clause[len(prefix):])
                if _looks_like_valid_name(name):
                    _add_unique_op(mem_ops, seen_mem, ("user", f"name: {name}"))
                break

            user_field = _extract_user_field_from_clause(clause)
            if user_field:
                key, value = user_field
                _add_unique_op(mem_ops, seen_mem, ("user", f"{key}: {value}"))

            for prefix, mapped in _PREFERENCE_PREFIXES:
                if not lowered.startswith(prefix):
                    continue
                value = _clean_extracted_value(clause[len(prefix):])
                if value:
                    _add_unique_op(mem_ops, seen_mem, ("preferences", f"{mapped}{value.lower()}"))
                break

            for prefix, mapped in _FACT_PREFIXES:
                if not lowered.startswith(prefix):
                    continue
                value = _clean_extracted_value(clause[len(prefix):])
                if value:
                    _add_unique_op(mem_ops, seen_mem, ("facts", f"{mapped}{value.lower()}"))
                break

            for prefix in _PROJECT_PREFIXES:
                if not lowered.startswith(prefix):
                    continue
                raw_project = _clean_extracted_value(clause[len(prefix):]).lower()
                if not raw_project:
                    break
                name = raw_project
                detail = ""
                for connector in (" using ", " with ", " for ", " on ", " at "):
                    if connector in name:
                        left, right = name.split(connector, 1)
                        name = _clean_extracted_value(left).lower()
                        detail = _clean_extracted_value(right).lower()
                        break
                if name:
                    _add_unique_op(project_ops, seen_projects, (name, detail))
                break

    return mem_ops, project_ops


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
            content = _normalize_entry_text(content)
            if mem_cat == "user":
                if ":" in content:
                    key, value = content.split(":", 1)
                    key = key.strip()
                    norm_value = _normalize_user_field(key, value)
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
            if detail and detail not in activity["details"] and not _looks_like_generic_memory(detail):
                activity["details"].append(detail)
                activity["updated"] = _now()

    store.update(update)
    store._flush(force=True)
    return True


def remember_user_message(user_text: str) -> bool:
    """Infer durable memory from a raw user message and persist it."""
    mem_ops, project_ops = _extract_memory_ops_from_user_text(user_text)
    if not mem_ops and not project_ops:
        return False
    return apply_tag_operations(
        mem_ops=mem_ops,
        forget_queries=[],
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
