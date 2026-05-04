"""Persistent user memory system with asyncio-driven writes and in-memory caching."""

import asyncio
import copy
import math
import re
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import orjson

from app.config import MEMORY_FLUSH_INTERVAL

MEMORY_PATH = Path(__file__).parent / "memory.json"

DEFAULT_MEMORY = {
    "user": {},
    "preferences": [],
    "facts": [],
    "history": [],
    "activities": {},
    "metadata": {
        "first_seen": None,
        "last_seen": None,
        "interaction_count": 0,
    }
}

_ENTRY_LIMITS = {
    "preferences": 24,
    "facts": 32,
    "history": 20,
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


def _fresh_default_memory() -> dict:
    return copy.deepcopy(DEFAULT_MEMORY)


def _ensure_memory_shape(data: dict) -> dict:
    """Normalize loaded memory data to the current schema."""
    normalized = _fresh_default_memory()
    if not isinstance(data, dict):
        return normalized

    for key in ("user", "activities", "metadata"):
        if isinstance(data.get(key), dict):
            normalized[key].update(copy.deepcopy(data[key]))

    for key in ("preferences", "facts", "history"):
        if isinstance(data.get(key), list):
            normalized[key] = copy.deepcopy(data[key])

    return normalized


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
    return re.sub(r"(^|[\s'-])([a-z])", lambda m: f"{m.group(1)}{m.group(2).upper()}", value)


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
        for token in _tokenize_embedding_text(_normalize_entry_text(value))
        if len(token) > 2 and token.lower() not in ENTITY_EMBEDDING_STOPWORDS
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
        return len(tokens) >= 3 and (
            lowered.startswith(("uses ", "has ", "runs ", "works on ", "working on "))
            or any(char.isdigit() for char in normalized)
        )
    if category == "history":
        return len(tokens) >= 3
    return True


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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


# ---------------------------------------------------------------------------
# Lightweight embedding helpers — used by search only
# ---------------------------------------------------------------------------

ENTITY_EMBEDDING_DIM = 96
SEARCH_SEMANTIC_THRESHOLD = 0.18
ENTITY_EMBEDDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "i", "if",
    "in", "into", "is", "it", "me", "my", "of", "on", "or", "our", "so", "that", "the", "this",
    "to", "was", "we", "with", "you", "your",
}
_PUNCT_TRANSLATION = str.maketrans({
    ch: " "
    for ch in "\"'`.,:;!?()[]{}<>|/\\@#$%^&*=+~"
})


def _tokenize_embedding_text(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").translate(_PUNCT_TRANSLATION).split())
    if not cleaned:
        return []
    return cleaned.split()


def _stable_feature_index(feature: str, *, salt: str) -> int:
    total = 0
    for idx, char in enumerate(f"{salt}:{feature}"):
        total = (total * 131 + ord(char) + idx) % 2_147_483_647
    return total % ENTITY_EMBEDDING_DIM


def _stable_feature_sign(feature: str, *, salt: str) -> float:
    total = 0
    for idx, char in enumerate(f"{feature}:{salt}"):
        total = (total * 137 + ord(char) + idx) % 65_537
    return 1.0 if total % 2 == 0 else -1.0


def _token_features(token: str) -> list[str]:
    lowered = token.lower()
    if not lowered:
        return []
    features = [f"tok:{lowered}"]
    padded = f"^{lowered}$"
    if len(padded) >= 4:
        for i in range(len(padded) - 2):
            features.append(f"tri:{padded[i:i + 3]}")
    return features


@lru_cache(maxsize=8192)
def _embed_text(text: str) -> tuple[float, ...]:
    tokens = [token.lower() for token in _tokenize_embedding_text(text)]
    if not tokens:
        return tuple(0.0 for _ in range(ENTITY_EMBEDDING_DIM))

    vector = [0.0] * ENTITY_EMBEDDING_DIM
    for token in tokens:
        if token in ENTITY_EMBEDDING_STOPWORDS:
            continue
        weight = 1.25 if any(char.isdigit() for char in token) else 1.0
        if any(char.isupper() for char in token):
            weight += 0.1
        for feature in _token_features(token):
            index = _stable_feature_index(feature, salt="idx")
            sign = _stable_feature_sign(feature, salt="sign")
            vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-8:
        return tuple(0.0 for _ in range(ENTITY_EMBEDDING_DIM))
    return tuple(value / norm for value in vector)


def _cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    return sum(left * right for left, right in zip(vec_a, vec_b))


# ---------------------------------------------------------------------------
# MemoryStore — thread-safe singleton with asyncio background flusher
# ---------------------------------------------------------------------------

class MemoryStore:
    """Thread-safe memory store with asyncio-driven background writes."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, path: Path = MEMORY_PATH, flush_interval: float = MEMORY_FLUSH_INTERVAL):
        self.path = path
        self.flush_interval = flush_interval
        self.data = None
        self.dirty = False
        self._flush_lock = threading.RLock()
        self._background_thread = None
        self._background_loop = None
        self._shutdown_event = None
        self._prompt_cache = None
        self._prompt_cache_gen = -1
        self._load_initial()
        self._start_background_flusher()

    @classmethod
    def get_instance(cls) -> "MemoryStore":
        """Singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

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
        else:
            self.data = _fresh_default_memory()

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

    def update_incremental(self, updater_func: Callable[[dict], None], *, invalidate_prompt_cache: bool = True) -> None:
        """Alias of update() — kept for call-site compatibility."""
        return self.update(updater_func, invalidate_prompt_cache=invalidate_prompt_cache)

    def _flush(self, force: bool = False) -> bool:
        """Write memory to disk if dirty or forced. Returns True if flushed."""
        with self._flush_lock:
            if not self.dirty and not force:
                return False
            try:
                temp_path = self.path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
                temp_path.replace(self.path)
                self.dirty = False
                return True
            except Exception as e:
                print(f"Error flushing memory: {e}")
                return False

    async def _background_flush_loop(self) -> None:
        """Periodically flush dirty memory until shutdown is requested."""
        while True:
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.flush_interval)
                break
            except asyncio.TimeoutError:
                self._flush()
        self._flush(force=True)

    def _run_background_flusher(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._background_loop = loop
        self._shutdown_event = asyncio.Event()
        try:
            loop.run_until_complete(self._background_flush_loop())
        finally:
            self._shutdown_event = None
            self._background_loop = None
            asyncio.set_event_loop(None)
            loop.close()

    def _start_background_flusher(self):
        """Start the asyncio background flusher on its own loop thread."""
        self._background_thread = threading.Thread(
            target=self._run_background_flusher,
            daemon=True,
            name="MemoryFlusher",
        )
        self._background_thread.start()

    def shutdown(self):
        """Gracefully stop background flusher and flush remaining data."""
        if self._background_loop and self._shutdown_event:
            self._background_loop.call_soon_threadsafe(self._shutdown_event.set)
        if self._background_thread:
            self._background_thread.join(timeout=2.0)
        self._flush(force=True)


# Singleton instance
_store = None


def get_store() -> MemoryStore:
    """Get the global memory store instance."""
    global _store
    if _store is None:
        _store = MemoryStore.get_instance()
    return _store


def reload_from_disk() -> None:
    """Refresh the singleton memory store from the current on-disk JSON file."""
    get_store().reload()


def flush_now() -> bool:
    """Synchronously flush pending in-memory changes to disk."""
    return get_store()._flush(force=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_tag_operations(
    *,
    mem_ops: list[tuple[str, str]],
    forget_queries: list[str],
    project_ops: list[tuple[str, str]],
    observe_ops: list = None,  # kept in signature for call-site compatibility, ignored
) -> bool:
    """Apply parsed memory tag operations in one authoritative place."""
    if not (mem_ops or forget_queries or project_ops):
        return False

    store = get_store()

    def update(data):
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
            for category in ("preferences", "facts", "history"):
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

    store.update_incremental(update)
    flush_now()
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_interaction() -> None:
    """Record that an interaction occurred and update basic metadata."""
    store = get_store()

    def update(data):
        meta = data["metadata"]
        now = _now()
        if meta["first_seen"] is None:
            meta["first_seen"] = now
        meta["last_seen"] = now
        meta["interaction_count"] += 1

    store.update_incremental(update)


def has_memory(category: str, content: str) -> bool:
    """Check if a memory exists."""
    data = get_store().get()
    content_lower = content.lower()

    if category == "user":
        for key, val in data["user"].items():
            if key.lower() in content_lower or content_lower in f"{key}: {val}".lower():
                return True
        return False

    if category == "activities":
        return any(content_lower in name for name in data.get("activities", {}))

    if category in ("preferences", "facts", "history"):
        return any(
            content_lower in e.get("content", "").lower()
            or e.get("content", "").lower() in content_lower
            for e in data[category]
        )

    return False


def add(category: str, content: str, weight: str = None) -> dict:
    """Add a memory entry."""
    store = get_store()
    content = _normalize_entry_text(content)

    if category not in ("user", "activities") and not _memory_entry_allowed(category, content):
        return {"success": False, "category": category, "ignored": True}

    def update(data):
        nonlocal category, content

        if category == "user":
            if isinstance(content, str):
                if ":" in content:
                    key, val = content.split(":", 1)
                    key = key.strip()
                    norm_val = _normalize_user_field(key, val)
                    data["user"][key] = norm_val
                else:
                    raise ValueError("user memories must be 'key: value' format")
            elif isinstance(content, dict):
                for key, val in content.items():
                    norm_val = _normalize_user_field(str(key), str(val))
                    data["user"][str(key)] = norm_val

        elif category in ("preferences", "facts", "history"):
            if not _touch_list_entry(data[category], content):
                data[category].append({"content": content, "created": _now(), "mentions": 1})
            _enforce_entry_limit(data, category)

        elif category == "activities":
            if ":" in content:
                name, detail = content.split(":", 1)
                name = name.strip().lower()
                detail = detail.strip()
                if name not in data["activities"]:
                    data["activities"][name] = {"details": [], "status": "active", "created": _now()}
                if detail and detail not in data["activities"][name]["details"] and not _looks_like_generic_memory(detail):
                    data["activities"][name]["details"].append(detail)
                    data["activities"][name]["updated"] = _now()
            else:
                name = content.strip().lower()
                if name not in data["activities"]:
                    data["activities"][name] = {"details": [], "status": "active", "created": _now()}
        else:
            raise ValueError(f"Unknown category: {category}")

    store.update_incremental(update)
    return {"success": True, "category": category}


def touch(category: str, content: str) -> None:
    """Increment mention count for existing memory or create it."""
    store = get_store()

    def update(data):
        nonlocal category, content
        if category not in data or not isinstance(data[category], list):
            return
        for entry in data[category]:
            existing = entry.get("content", "").lower()
            if existing == content.lower() or content.lower() in existing or existing in content.lower():
                entry["mentions"] = entry.get("mentions", 1) + 1
                entry["last_seen"] = _now()
                return
        add(category, content)

    store.update_incremental(update)


def forget(query: str) -> dict:
    """Remove memories matching a query."""
    store = get_store()
    removed = []
    query_lower = query.lower()

    def update(data):
        nonlocal removed, query_lower

        for category in ("preferences", "facts", "history"):
            kept_entries = []
            for entry in data[category]:
                if query_lower in entry.get("content", "").lower():
                    removed.append({"category": category, "content": entry["content"]})
                    continue
                kept_entries.append(entry)
            data[category] = kept_entries

        keys_to_remove = [k for k, v in data["user"].items()
                          if query_lower in k.lower() or query_lower in str(v).lower()]
        for k in keys_to_remove:
            removed.append({"category": "user", "key": k, "value": data["user"][k]})
            del data["user"][k]

        to_remove = [
            name for name, activity in data["activities"].items()
            if query_lower in name or any(query_lower in d.lower() for d in activity.get("details", []))
        ]
        for name in to_remove:
            removed.append({"category": "activity", "name": name, **data["activities"][name]})
            del data["activities"][name]

    store.update_incremental(update)
    return {"removed": removed, "count": len(removed)}


def archive_activity(name: str) -> dict:
    """Mark a project as done."""
    store = get_store()
    name = name.lower()

    def update(data):
        nonlocal name
        if name in data["activities"]:
            data["activities"][name]["status"] = "done"
            data["activities"][name]["completed"] = _now()
            return {"success": True, "project": name}
        return {"error": f"Project '{name}' not found"}

    result = store.update_incremental(update)
    return result if result else {"error": f"Project '{name}' not found"}


def delete(category: str, index: int = None, key: str = None) -> dict:
    """Delete a memory entry by index or key."""
    store = get_store()

    def update(data):
        nonlocal category, index, key

        if category == "user" and key:
            if key in data["user"]:
                del data["user"][key]
                return {"success": True}
            return {"error": f"Key '{key}' not found"}

        if category in ("preferences", "facts", "history"):
            if index is not None and 0 <= index < len(data[category]):
                removed = data[category].pop(index)
                return {"success": True, "removed": removed}
            return {"error": f"Invalid index {index}"}

        return {"error": f"Unknown category: {category}"}

    result = store.update_incremental(update)
    return result if result else {"error": "Update failed"}


def _memory_search_documents(data: dict) -> list[dict]:
    documents: list[dict] = []

    for key, val in data.get("user", {}).items():
        documents.append({
            "payload": {"category": "user", "key": key, "value": val},
            "text": f"{key} {val}",
        })

    for category in ("preferences", "facts", "history"):
        for i, entry in enumerate(data.get(category, [])):
            documents.append({
                "payload": {"category": category, "index": i, **entry},
                "text": entry.get("content", ""),
            })

    for name, act in data.get("activities", {}).items():
        documents.append({
            "payload": {"category": "activity", "name": name, **act},
            "text": " ".join([name, *act.get("details", [])]),
        })

    return documents


def search(query: str) -> list[dict]:
    """Search memories using lightweight semantic embeddings."""
    data = get_store().get()
    query_text = " ".join(str(query or "").split())
    if not query_text:
        return []

    query_vector = _embed_text(query_text)
    query_tokens = {token.lower() for token in _tokenize_embedding_text(query_text)}
    scored: list[tuple[float, dict]] = []

    for doc in _memory_search_documents(data):
        text = " ".join(str(doc.get("text", "")).split())
        if not text:
            continue
        score = _cosine_similarity(query_vector, _embed_text(text))
        doc_tokens = {token.lower() for token in _tokenize_embedding_text(text)}
        overlap = query_tokens & doc_tokens
        if overlap:
            score += min(0.18, 0.05 * len(overlap))
        if score < SEARCH_SEMANTIC_THRESHOLD:
            continue
        payload = dict(doc["payload"])
        payload["score"] = round(score, 4)
        scored.append((score, payload))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored[:12]]


def get_all() -> dict:
    """Get all memory data."""
    return get_store().get()


def format_for_prompt() -> str:
    """Format all memories as a string for the system prompt."""
    store = get_store()
    with store._flush_lock:
        data = store.data
        cache_gen = store._prompt_cache_gen

    current_gen = (cache_gen, store.data["metadata"]["interaction_count"])
    if store._prompt_cache is not None and store._prompt_cache_gen == current_gen[0]:
        return store._prompt_cache

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