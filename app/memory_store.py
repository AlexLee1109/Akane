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
    "observations": [],
    "activities": {},
    "entities": {},
    "graph": {
        "edges": [],
    },
    "metadata": {
        "first_seen": None,
        "last_seen": None,
        "interaction_count": 0,
        "mood_trend": [],
    }
}

NEUTRAL = "neutral"
POSITIVE = "positive"
NEGATIVE = "negative"

_ENTRY_LIMITS = {
    "preferences": 24,
    "facts": 32,
    "history": 20,
    "observations": 24,
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

    for key in ("user", "activities", "metadata", "entities", "graph"):
        if isinstance(data.get(key), dict):
            normalized[key].update(copy.deepcopy(data[key]))

    for key in ("preferences", "facts", "history", "observations"):
        if isinstance(data.get(key), list):
            normalized[key] = copy.deepcopy(data[key])

    if not isinstance(normalized["graph"].get("edges"), list):
        normalized["graph"]["edges"] = []

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


def _capitalize_name_piece(piece: str) -> str:
    if not piece:
        return piece
    return piece[:1].upper() + piece[1:].lower()


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


def _memory_entry_allowed(category: str, content: str, *, weight: str = NEUTRAL) -> bool:
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
    if category == "observations":
        return weight != NEUTRAL and len(tokens) >= 2
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
    kept: list[dict] = []
    for entry in entries:
        if id(entry) in keep_ids:
            kept.append(entry)
            continue
        if category in {"preferences", "facts", "observations"}:
            _release_graph_entry(data, category, entry)
    data[category] = kept


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def _ensure_graph_shape(data: dict) -> None:
    if not isinstance(data.get("entities"), dict):
        data["entities"] = {}
    if not isinstance(data.get("graph"), dict):
        data["graph"] = {"edges": [], "_edge_set": set()}
    if not isinstance(data["graph"].get("edges"), list):
        data["graph"]["edges"] = []
    # Ensure the edge set is always present and in sync after a load
    if not isinstance(data["graph"].get("_edge_set"), set):
        data["graph"]["_edge_set"] = {
            (e["source"], e["relation"], e["target"])
            for e in data["graph"]["edges"]
        }


def _entity_display_name(entity_type: str, name: str) -> str:
    cleaned = " ".join(str(name).strip().split())
    if not cleaned:
        return cleaned
    if entity_type in {"user", "person"}:
        return _normalize_person_name(cleaned)
    if cleaned.lower() == "macos":
        return "macOS"
    if cleaned.lower() == "python":
        return "Python"
    if cleaned.lower() == "pytorch":
        return "PyTorch"
    if cleaned.lower() in {"vscode", "vs code"}:
        return "VS Code"
    return cleaned


def _infer_entity_type(name: str) -> str:
    entity_type, _ = _infer_entity_type_semantic(name)
    return entity_type


def _looks_like_person_entity(value: str) -> bool:
    candidate = value.strip()
    if not candidate or len(candidate) > 60:
        return False
    if candidate == candidate.lower():
        return False
    parts = candidate.replace("-", " ").replace("'", " ").split()
    if not 1 <= len(parts) <= 4:
        return False
    return all(part.isalpha() for part in parts)


def _upsert_entity(
    data: dict,
    entity_type: str,
    name: str,
    *,
    explicit_id: str | None = None,
    aliases: list[str] | None = None,
    attrs: dict | None = None,
) -> str:
    _ensure_graph_shape(data)
    display_name = _entity_display_name(entity_type, name)
    if not display_name:
        return ""

    entity_id = explicit_id or f"{entity_type}:{_slugify(display_name)}"
    entity = data["entities"].get(entity_id)
    now = _now()
    if entity is None:
        entity = {
            "id": entity_id,
            "type": entity_type,
            "name": display_name,
            "aliases": [],
            "mentions": 0,
            "created": now,
            "last_seen": now,
            "attrs": {},
        }
        data["entities"][entity_id] = entity

    entity["mentions"] = entity.get("mentions", 0) + 1
    entity["last_seen"] = now
    if explicit_id or len(display_name) > len(entity.get("name", "")):
        entity["name"] = display_name

    alias_list = entity.setdefault("aliases", [])
    for alias in aliases or []:
        normalized = " ".join(str(alias).strip().split())
        if normalized and normalized != entity["name"] and normalized not in alias_list:
            alias_list.append(normalized)

    attr_dict = entity.setdefault("attrs", {})
    for key, value in (attrs or {}).items():
        if value is not None and value != "":
            attr_dict[key] = value

    return entity_id


def _touch_edge(data: dict, source: str, relation: str, target: str, *, attrs: dict | None = None) -> None:
    """Add or update an edge, using an in-memory set for O(1) existence checks."""
    if not source or not target:
        return
    _ensure_graph_shape(data)
    now = _now()
    edge_key = (source, relation, target)

    if edge_key in data["graph"]["_edge_set"]:
        # Edge already exists — find it and update in-place.
        for edge in data["graph"]["edges"]:
            if edge["source"] == source and edge["relation"] == relation and edge["target"] == target:
                edge["mentions"] = edge.get("mentions", 0) + 1
                edge["last_seen"] = now
                if attrs:
                    edge.setdefault("attrs", {}).update(
                        {k: v for k, v in attrs.items() if v not in (None, "")}
                    )
                return
    else:
        data["graph"]["_edge_set"].add(edge_key)
        data["graph"]["edges"].append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "mentions": 1,
                "created": now,
                "last_seen": now,
                "attrs": {k: v for k, v in (attrs or {}).items() if v not in (None, "")},
            }
        )


ENTITY_IGNORE_WORDS = {
    "using", "use", "with", "code", "memory", "graph", "user", "project",
    "activity", "detail", "fact", "preference", "observation",
}
ENTITY_EMBEDDING_DIM = 96
ENTITY_SEMANTIC_THRESHOLD = 0.34
SEARCH_SEMANTIC_THRESHOLD = 0.18
ENTITY_TYPE_SEEDS = {
    "platform": {
        "macos", "windows", "linux", "ubuntu", "debian", "colab", "desktop", "laptop",
    },
    "hardware": {
        "gpu", "cpu", "a100", "h100", "rtx", "4090", "3090", "96gb", "80gb", "40gb", "vram",
    },
    "software": {
        "python", "pytorch", "cuda", "mps", "vscode", "vs code", "llama", "gguf", "bf16",
        "bfloat16", "fp16", "javascript", "typescript",
    },
    "person": {
        "arcane", "alex", "jordan", "sam", "neko", "ayaka", "developer", "creator", "friend",
    },
    "topic": {
        "memory", "graph", "project", "training", "prompt", "model", "system",
    },
}
ENTITY_SEED_TOKENS = {
    token
    for seeds in ENTITY_TYPE_SEEDS.values()
    for seed in seeds
    for token in seed.split()
}
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


@lru_cache(maxsize=256)
def _entity_type_prototypes() -> dict[str, tuple[float, ...]]:
    return {
        entity_type: _embed_text(" ".join(sorted(seeds)))
        for entity_type, seeds in ENTITY_TYPE_SEEDS.items()
    }


def _candidate_entity_spans(text: str, *, max_words: int = 4) -> list[str]:
    words = _tokenize_embedding_text(text)
    if not words:
        return []

    spans: list[str] = []
    seen: set[str] = set()
    for start in range(len(words)):
        for width in range(1, max_words + 1):
            end = start + width
            if end > len(words):
                break
            chunk_words = words[start:end]
            normalized = " ".join(chunk_words).strip()
            lowered = normalized.lower()
            if not lowered or lowered in seen:
                continue
            if chunk_words[0].lower() in ENTITY_EMBEDDING_STOPWORDS or chunk_words[-1].lower() in ENTITY_EMBEDDING_STOPWORDS:
                continue
            if all(token.lower() in ENTITY_EMBEDDING_STOPWORDS for token in chunk_words):
                continue
            if width > 1 and any(token.lower() in ENTITY_EMBEDDING_STOPWORDS for token in chunk_words[1:-1]):
                continue
            if width == 1 and len(lowered) < 3:
                continue
            if width > 2:
                informative_tokens = 0
                for token in chunk_words:
                    token_lower = token.lower()
                    if (
                        token_lower in ENTITY_SEED_TOKENS
                        or any(char.isdigit() for char in token)
                        or token[:1].isupper()
                    ):
                        informative_tokens += 1
                if informative_tokens < width - 1:
                    continue
            seen.add(lowered)
            spans.append(normalized)
    return spans


def _seed_overlap_score(candidate: str, entity_type: str) -> float:
    candidate_tokens = {token.lower() for token in _tokenize_embedding_text(candidate)}
    if not candidate_tokens:
        return 0.0
    overlap = candidate_tokens & ENTITY_TYPE_SEEDS.get(entity_type, set())
    if not overlap:
        return 0.0
    return min(0.25, 0.08 * len(overlap))


def _infer_entity_type_semantic(candidate: str) -> tuple[str, float]:
    vector = _embed_text(candidate)
    prototypes = _entity_type_prototypes()
    best_type = "topic"
    best_score = -1.0
    for entity_type, prototype in prototypes.items():
        score = _cosine_similarity(vector, prototype) + _seed_overlap_score(candidate, entity_type)
        if entity_type == "person" and not any(part[:1].isupper() for part in candidate.split()):
            score -= 0.12
        if score > best_score:
            best_type = entity_type
            best_score = score
    return best_type, best_score


def _looks_entity_like(candidate: str, score: float) -> bool:
    lowered = candidate.lower().strip()
    if not lowered or lowered in ENTITY_IGNORE_WORDS:
        return False
    if len(lowered) < 3:
        return False
    tokens = candidate.split()
    if tokens[0].lower() in ENTITY_EMBEDDING_STOPWORDS or tokens[-1].lower() in ENTITY_EMBEDDING_STOPWORDS:
        return False
    if any(char.isdigit() for char in lowered):
        return True
    if any(token.lower() in ENTITY_SEED_TOKENS for token in tokens):
        return True
    if any(part[:1].isupper() for part in tokens) and len(tokens) <= 3:
        return True
    return score >= ENTITY_SEMANTIC_THRESHOLD


@lru_cache(maxsize=2048)
def _cached_entity_mentions(text: str) -> tuple[tuple[str, str], ...]:
    mentions: list[tuple[float, str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not text:
        return ()

    for candidate in _candidate_entity_spans(text):
        entity_type, score = _infer_entity_type_semantic(candidate)
        if not _looks_entity_like(candidate, score):
            continue
        item = (entity_type, _entity_display_name(entity_type, candidate))
        if item not in seen:
            seen.add(item)
            mentions.append((score, item[0], item[1]))

    mentions.sort(key=lambda item: (-item[0], len(item[2])))
    chosen: list[tuple[str, str]] = []
    chosen_names: list[str] = []
    for _, entity_type, display_name in mentions:
        lowered = display_name.lower()
        if any(lowered in existing or existing in lowered for existing in chosen_names):
            continue
        chosen_names.append(lowered)
        chosen.append((entity_type, display_name))

    return tuple(chosen)


def _extract_entity_mentions(text: str) -> list[tuple[str, str]]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    return list(_cached_entity_mentions(normalized))


def _parse_relation_content(content: str) -> tuple[str, str] | None:
    text = " ".join(content.strip().split())
    if not text:
        return None

    patterns = (
        ("doesn't want ", "does_not_want"),
        ("doesn't like ", "does_not_like"),
        ("prefers ", "prefers"),
        ("likes ", "likes"),
        ("loves ", "likes"),
        ("enjoys ", "likes"),
        ("wants ", "wants"),
        ("needs ", "needs"),
        ("uses ", "uses"),
        ("using ", "uses"),
        ("has ", "has"),
        ("on ", "uses"),
    )
    lowered = text.lower()
    for prefix, relation in patterns:
        if lowered.startswith(prefix):
            return relation, text[len(prefix):].strip()
    return None


# ---------------------------------------------------------------------------
# Graph indexing helpers — used by both the full rebuild and incremental path
# ---------------------------------------------------------------------------

def _index_user_fields(data: dict, user_id: str) -> None:
    """Index all user profile fields into the graph."""
    for key, value in data.get("user", {}).items():
        if key == "name":
            _upsert_entity(data, "user", value, explicit_id="user:self", aliases=[value])
            continue
        attribute_id = _upsert_entity(data, "attribute", f"{key}: {value}")
        _touch_edge(data, user_id, "has_attribute", attribute_id, attrs={"field": key})
        for entity_type, entity_name in _extract_entity_mentions(str(value)):
            mention_id = _upsert_entity(data, entity_type, entity_name)
            _touch_edge(data, attribute_id, "mentions", mention_id, attrs={"field": key})


def _index_entry(data: dict, section: str, entry: dict, user_id: str) -> None:
    """Index a single list entry (preference / fact / observation) into the graph."""
    content = entry.get("content", "").strip()
    if not content:
        return

    if section == "preferences":
        node_id = _upsert_entity(data, "preference", content)
        _touch_edge(data, user_id, "has_preference", node_id)
        parsed = _parse_relation_content(content)
        if parsed:
            relation, obj = parsed
            object_id = _upsert_entity(data, _infer_entity_type(obj), obj)
            _touch_edge(data, user_id, relation, object_id)
            _touch_edge(data, node_id, "about", object_id)
        for entity_type, entity_name in _extract_entity_mentions(content):
            mention_id = _upsert_entity(data, entity_type, entity_name)
            _touch_edge(data, node_id, "mentions", mention_id)

    elif section == "facts":
        node_id = _upsert_entity(data, "fact", content)
        _touch_edge(data, user_id, "has_fact", node_id)
        parsed = _parse_relation_content(content)
        if parsed:
            relation, obj = parsed
            object_id = _upsert_entity(data, _infer_entity_type(obj), obj)
            _touch_edge(data, user_id, relation, object_id)
            _touch_edge(data, node_id, "about", object_id)
        for entity_type, entity_name in _extract_entity_mentions(content):
            mention_id = _upsert_entity(data, entity_type, entity_name)
            _touch_edge(data, node_id, "mentions", mention_id)

    elif section == "observations":
        node_id = _upsert_entity(
            data,
            "observation",
            content,
            attrs={"weight": entry.get("weight", NEUTRAL)},
        )
        _touch_edge(data, user_id, "observed", node_id)
        for entity_type, entity_name in _extract_entity_mentions(content):
            mention_id = _upsert_entity(data, entity_type, entity_name)
            _touch_edge(data, node_id, "mentions", mention_id)


def _index_activity(data: dict, name: str, activity: dict, user_id: str) -> None:
    """Index a single activity/project into the graph."""
    project_id = _upsert_entity(
        data,
        "project",
        name,
        attrs={"status": activity.get("status", "active")},
    )
    status = activity.get("status", "active")
    relation = "working_on" if status == "active" else "completed" if status == "done" else "paused"
    _touch_edge(data, user_id, relation, project_id)
    for detail in activity.get("details", []):
        detail_id = _upsert_entity(data, "detail", detail)
        _touch_edge(data, project_id, "has_detail", detail_id)
        for entity_type, entity_name in _extract_entity_mentions(detail):
            mention_id = _upsert_entity(data, entity_type, entity_name)
            _touch_edge(data, detail_id, "mentions", mention_id)


def _rebuild_graph(data: dict) -> None:
    """Full graph rebuild from scratch. Only called on load or after bulk mutations."""
    _ensure_graph_shape(data)
    data["entities"] = {}
    data["graph"]["edges"] = []
    data["graph"]["_edge_set"] = set()

    user_name = data.get("user", {}).get("name") or "User"
    user_id = _upsert_entity(
        data,
        "user",
        user_name,
        explicit_id="user:self",
        attrs={"profile_fields": len(data.get("user", {}))},
    )

    _index_user_fields(data, user_id)

    for section in ("preferences", "facts", "observations"):
        for entry in data.get(section, []):
            _index_entry(data, section, entry, user_id)

    for name, activity in data.get("activities", {}).items():
        _index_activity(data, name, activity, user_id)


def _patch_graph_entry(data: dict, section: str, entry: dict) -> None:
    """
    Incrementally add a single new entry to an already-current graph.
    Avoids a full rebuild for the common case of appending one memory.
    """
    _ensure_graph_shape(data)
    user_id = "user:self"
    if section in ("preferences", "facts", "observations"):
        _index_entry(data, section, entry, user_id)
    # User-field and activity incremental patches are handled inline in
    # apply_tag_operations / add(), which call _patch_graph_user_field and
    # _patch_graph_activity respectively.


def _patch_graph_user_field(data: dict, key: str, value: str) -> None:
    """Incrementally update graph for a single user profile field change."""
    _ensure_graph_shape(data)
    user_id = "user:self"
    if key == "name":
        _upsert_entity(data, "user", value, explicit_id="user:self", aliases=[value])
        return
    attribute_id = _upsert_entity(data, "attribute", f"{key}: {value}")
    _touch_edge(data, user_id, "has_attribute", attribute_id, attrs={"field": key})
    for entity_type, entity_name in _extract_entity_mentions(str(value)):
        mention_id = _upsert_entity(data, entity_type, entity_name)
        _touch_edge(data, attribute_id, "mentions", mention_id, attrs={"field": key})


def _patch_graph_activity(data: dict, name: str) -> None:
    """Incrementally update graph for a single activity after it has been mutated."""
    activity = data.get("activities", {}).get(name)
    if activity is None:
        return
    _index_activity(data, name, activity, "user:self")


def _entry_node_type(section: str) -> str:
    return {
        "preferences": "preference",
        "facts": "fact",
        "observations": "observation",
    }.get(section, section.rstrip("s"))


def _entry_primary_relation(section: str) -> str:
    return {
        "preferences": "has_preference",
        "facts": "has_fact",
        "observations": "observed",
    }[section]


def _entity_id_for(entity_type: str, name: str) -> str:
    display_name = _entity_display_name(entity_type, name)
    if not display_name:
        return ""
    return f"{entity_type}:{_slugify(display_name)}"


def _release_edge(data: dict, source: str, relation: str, target: str) -> bool:
    """Decrement an edge mention count or remove it when it hits zero."""
    if not source or not target:
        return False
    _ensure_graph_shape(data)
    edges = data["graph"]["edges"]
    edge_key = (source, relation, target)
    for index, edge in enumerate(edges):
        if edge["source"] != source or edge["relation"] != relation or edge["target"] != target:
            continue
        mentions = int(edge.get("mentions", 1)) - 1
        if mentions > 0:
            edge["mentions"] = mentions
        else:
            edges.pop(index)
            data["graph"]["_edge_set"].discard(edge_key)
        return True
    return False


def _entity_has_references(data: dict, entity_id: str) -> bool:
    return any(
        edge["source"] == entity_id or edge["target"] == entity_id
        for edge in data.get("graph", {}).get("edges", [])
    )


def _release_entity(data: dict, entity_id: str, *, protected: set[str] | None = None) -> bool:
    """Decrement entity mentions and drop the node once it becomes orphaned."""
    if not entity_id:
        return False
    protected = protected or {"user:self"}
    if entity_id in protected:
        return False
    entity = data.get("entities", {}).get(entity_id)
    if entity is None:
        return False
    mentions = int(entity.get("mentions", 1)) - 1
    entity["mentions"] = max(mentions, 0)
    if entity["mentions"] <= 0 and not _entity_has_references(data, entity_id):
        del data["entities"][entity_id]
        return True
    return False


def _prune_orphan_entities(data: dict, candidate_ids: list[str]) -> None:
    """Remove candidate entities that no longer participate in any edge."""
    pending = [entity_id for entity_id in candidate_ids if entity_id and entity_id != "user:self"]
    seen: set[str] = set()
    while pending:
        entity_id = pending.pop()
        if entity_id in seen:
            continue
        seen.add(entity_id)
        entity = data.get("entities", {}).get(entity_id)
        if entity is None:
            continue
        if _entity_has_references(data, entity_id):
            continue
        del data["entities"][entity_id]


def _release_entry_mentions(data: dict, source_id: str, text: str) -> list[str]:
    prunable: list[str] = []
    for entity_type, entity_name in _extract_entity_mentions(text):
        mention_id = _entity_id_for(entity_type, entity_name)
        if not mention_id:
            continue
        _release_edge(data, source_id, "mentions", mention_id)
        _release_entity(data, mention_id)
        prunable.append(mention_id)
    return prunable


def _release_graph_entry(data: dict, section: str, entry: dict) -> None:
    """Incrementally remove a single entry from the graph."""
    _ensure_graph_shape(data)
    user_id = "user:self"
    content = str(entry.get("content", "") or "").strip()
    if not content or section not in {"preferences", "facts", "observations"}:
        return

    node_type = _entry_node_type(section)
    node_id = _entity_id_for(node_type, content)
    if not node_id:
        return

    prunable: list[str] = [node_id]
    _release_edge(data, user_id, _entry_primary_relation(section), node_id)

    if section in {"preferences", "facts"}:
        parsed = _parse_relation_content(content)
        if parsed:
            _, obj = parsed
            object_id = _entity_id_for(_infer_entity_type(obj), obj)
            if object_id:
                _release_edge(data, node_id, "about", object_id)
                _release_entity(data, object_id)
                prunable.append(object_id)
        prunable.extend(_release_entry_mentions(data, node_id, content))
    elif section == "observations":
        prunable.extend(_release_entry_mentions(data, node_id, content))

    _release_entity(data, node_id)
    _prune_orphan_entities(data, prunable)


def _release_graph_user_field(data: dict, key: str, value: str) -> None:
    """Incrementally remove a user field from the graph."""
    _ensure_graph_shape(data)
    normalized_key = str(key or "").strip()
    normalized_value = str(value or "").strip()
    user_entity = data.get("entities", {}).get("user:self")
    if normalized_key == "name":
        if user_entity is not None:
            fallback_name = data.get("user", {}).get("name") or "User"
            user_entity["name"] = _entity_display_name("user", fallback_name)
            user_entity["aliases"] = [alias for alias in user_entity.get("aliases", []) if alias != normalized_value]
        return

    attribute_id = _entity_id_for("attribute", f"{normalized_key}: {normalized_value}")
    if not attribute_id:
        return

    prunable: list[str] = [attribute_id]
    _release_edge(data, "user:self", "has_attribute", attribute_id)
    for entity_type, entity_name in _extract_entity_mentions(normalized_value):
        mention_id = _entity_id_for(entity_type, entity_name)
        if not mention_id:
            continue
        _release_edge(data, attribute_id, "mentions", mention_id)
        _release_entity(data, mention_id)
        prunable.append(mention_id)
    _release_entity(data, attribute_id)
    _prune_orphan_entities(data, prunable)


def _release_graph_activity(data: dict, name: str, activity: dict) -> None:
    """Incrementally remove an activity/project from the graph."""
    _ensure_graph_shape(data)
    project_id = _entity_id_for("project", name)
    if not project_id:
        return

    status = activity.get("status", "active")
    relation = "working_on" if status == "active" else "completed" if status == "done" else "paused"
    prunable: list[str] = [project_id]

    _release_edge(data, "user:self", relation, project_id)
    for detail in activity.get("details", []):
        detail_id = _entity_id_for("detail", detail)
        if not detail_id:
            continue
        _release_edge(data, project_id, "has_detail", detail_id)
        for entity_type, entity_name in _extract_entity_mentions(detail):
            mention_id = _entity_id_for(entity_type, entity_name)
            if not mention_id:
                continue
            _release_edge(data, detail_id, "mentions", mention_id)
            _release_entity(data, mention_id)
            prunable.append(mention_id)
        _release_entity(data, detail_id)
        prunable.append(detail_id)

    _release_entity(data, project_id)
    _prune_orphan_entities(data, prunable)


class MemoryStore:
    """Thread-safe memory store with asyncio-driven background writes."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, path: Path = MEMORY_PATH, flush_interval: float = MEMORY_FLUSH_INTERVAL):
        self.path = path
        self.flush_interval = flush_interval
        self.data = None
        self.dirty = False
        self._graph_dirty = True
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
        self._graph_dirty = True
        self._ensure_graph_current_unlocked()

    def _ensure_graph_current_unlocked(self) -> None:
        """Run a full rebuild only if the graph is marked dirty."""
        if not self._graph_dirty:
            return
        _rebuild_graph(self.data)
        self._graph_dirty = False

    def reload(self) -> None:
        """Refresh the in-memory store from disk and invalidate prompt cache."""
        with self._flush_lock:
            if self.dirty:
                self._flush(force=True)
            self._reload_from_disk()
            self._prompt_cache = None
            self._prompt_cache_gen += 1

    def get(self) -> dict:
        """Get a deep copy of current memory state (graph excluded from copy)."""
        with self._flush_lock:
            self._ensure_graph_current_unlocked()
            snapshot = copy.deepcopy(self.data)
            # _edge_set is a transient index — strip it from snapshots so
            # callers don't accidentally serialise it.
            snapshot.get("graph", {}).pop("_edge_set", None)
            return snapshot

    def update(self, updater_func: Callable[[dict], None], *, invalidate_prompt_cache: bool = True) -> None:
        """
        Atomically update memory in-place and mark as dirty.
        The updater_func receives the memory dict and can modify it freely.
        Graph dirtying is opt-in via the returned sentinel; callers that
        perform incremental patches set _graph_dirty=False themselves.
        """
        with self._flush_lock:
            result = updater_func(self.data)
            self.dirty = True
            if invalidate_prompt_cache:
                self._prompt_cache_gen += 1
            return result

    def update_incremental(self, updater_func: Callable[[dict], None], *, invalidate_prompt_cache: bool = True) -> None:
        """
        Like update(), but promises the graph has already been patched
        incrementally inside updater_func — skips the full-rebuild flag.
        """
        with self._flush_lock:
            self._ensure_graph_current_unlocked()  # make sure graph is fresh first
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
                self._ensure_graph_current_unlocked()
                # Strip the transient edge set before serialising
                edges_backup = self.data["graph"].pop("_edge_set", None)
                temp_path = self.path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
                temp_path.replace(self.path)
                # Restore after write
                if edges_backup is not None:
                    self.data["graph"]["_edge_set"] = edges_backup
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
    observe_ops: list[str],
    forget_queries: list[str],
    project_ops: list[tuple[str, str]],
) -> bool:
    """Apply parsed memory tag operations in one authoritative place."""
    if not (mem_ops or observe_ops or forget_queries or project_ops):
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
                    _patch_graph_user_field(data, key, norm_value)
                continue
            if not _memory_entry_allowed(mem_cat, content):
                continue

            entry = None
            if not _touch_list_entry(data[mem_cat], content):
                entry = _new_list_entry(content)
                data[mem_cat].append(entry)

            # Patch the graph for new entries; touched entries already
            # have nodes — just let the mention count update stand.
            if entry is not None:
                _patch_graph_entry(data, mem_cat, entry)
            _enforce_entry_limit(data, mem_cat)

        for content, weight in observe_ops:
            content = _normalize_entry_text(content)
            if not _memory_entry_allowed("observations", content, weight=weight):
                continue
            found = False
            for obs in data["observations"]:
                if _entries_match(obs.get("content", ""), content):
                    obs["mentions"] = obs.get("mentions", 1) + 1
                    obs["last_seen"] = _now()
                    obs["weight"] = weight
                    found = True
                    break
            if not found:
                entry = {
                    "content": content,
                    "created": _now(),
                    "last_seen": _now(),
                    "mentions": 1,
                    "weight": weight,
                }
                data["observations"].append(entry)
                _patch_graph_entry(data, "observations", entry)
            _enforce_entry_limit(data, "observations")

        for query in forget_queries:
            query_lower = query.lower()
            for category in ("preferences", "facts", "history", "observations"):
                kept_entries = []
                for entry in data[category]:
                    if query_lower in entry.get("content", "").lower():
                        if category in {"preferences", "facts", "observations"}:
                            _release_graph_entry(data, category, entry)
                        continue
                    kept_entries.append(entry)
                data[category] = kept_entries

            keys_to_remove = [
                key for key, value in data["user"].items()
                if query_lower in key.lower() or query_lower in str(value).lower()
            ]
            for key in keys_to_remove:
                _release_graph_user_field(data, key, data["user"][key])
                del data["user"][key]

            activity_names = [
                name
                for name, activity in data["activities"].items()
                if query_lower in name or any(query_lower in detail.lower() for detail in activity.get("details", []))
            ]
            for name in activity_names:
                _release_graph_activity(data, name, data["activities"][name])
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
                    _release_graph_activity(data, name, activity)
                    del data["activities"][name]
                    continue

                _patch_graph_activity(data, name)
                continue

            if activity is None:
                activity = data["activities"][name] = {"details": [], "status": "active", "created": _now()}
            if detail and detail not in activity["details"] and not _looks_like_generic_memory(detail):
                activity["details"].append(detail)
                activity["updated"] = _now()
            _patch_graph_activity(data, name)

    store.update_incremental(update)

    flush_now()
    return True


# Compatibility functions that use the store
def record_interaction(conversation_context: dict = None) -> None:
    """Record that an interaction occurred and update basic metadata."""
    store = get_store()

    def update(data):
        meta = data["metadata"]
        now = _now()

        if meta["first_seen"] is None:
            meta["first_seen"] = now

        meta["last_seen"] = now
        meta["interaction_count"] += 1

    # Metadata updates don't touch the graph — use incremental path.
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

    if category in ("preferences", "facts", "history", "observations"):
        return any(
            content_lower in e.get("content", "").lower()
            or e.get("content", "").lower() in content_lower
            for e in data[category]
        )

    return False


def add(category: str, content: str, weight: str = NEUTRAL) -> dict:
    """Add a memory entry."""
    store = get_store()
    content = _normalize_entry_text(content)

    if category != "user" and category != "activities" and not _memory_entry_allowed(category, content, weight=weight):
        return {"success": False, "category": category, "ignored": True}

    def update(data):
        nonlocal category, content, weight

        if category == "user":
            if isinstance(content, str):
                if ":" in content:
                    key, val = content.split(":", 1)
                    key = key.strip()
                    norm_val = _normalize_user_field(key, val)
                    data["user"][key] = norm_val
                    _patch_graph_user_field(data, key, norm_val)
                else:
                    raise ValueError("user memories must be 'key: value' format")
            elif isinstance(content, dict):
                for key, val in content.items():
                    norm_val = _normalize_user_field(str(key), str(val))
                    data["user"][str(key)] = norm_val
                    _patch_graph_user_field(data, str(key), norm_val)

        elif category in ("preferences", "facts", "history"):
            if not _touch_list_entry(data[category], content):
                entry = {"content": content, "created": _now(), "mentions": 1}
                data[category].append(entry)
                _patch_graph_entry(data, category, entry)
            _enforce_entry_limit(data, category)

        elif category == "observations":
            if not _touch_list_entry(data["observations"], content):
                entry = {
                    "content": content,
                    "created": _now(),
                    "last_seen": _now(),
                    "mentions": 1,
                    "weight": weight,
                }
                data["observations"].append(entry)
                _patch_graph_entry(data, "observations", entry)
            _enforce_entry_limit(data, "observations")

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
            _patch_graph_activity(data, name)
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
    """Remove memories matching a query with incremental graph cleanup."""
    store = get_store()
    removed = []
    query_lower = query.lower()

    def update(data):
        nonlocal removed, query_lower

        for category in ("preferences", "facts", "history", "observations"):
            kept_entries = []
            for entry in data[category]:
                if query_lower in entry.get("content", "").lower():
                    removed.append({"category": category, "content": entry["content"]})
                    if category in {"preferences", "facts", "observations"}:
                        _release_graph_entry(data, category, entry)
                    continue
                kept_entries.append(entry)
            data[category] = kept_entries

        keys_to_remove = [k for k, v in data["user"].items()
                          if query_lower in k.lower() or query_lower in str(v).lower()]
        for k in keys_to_remove:
            removed.append({"category": "user", "key": k, "value": data["user"][k]})
            _release_graph_user_field(data, k, data["user"][k])
            del data["user"][k]

        to_remove = [
            name
            for name, activity in data["activities"].items()
            if query_lower in name or any(query_lower in detail.lower() for detail in activity.get("details", []))
        ]
        for name in to_remove:
            removed.append({"category": "activity", "name": name, **data["activities"][name]})
            _release_graph_activity(data, name, data["activities"][name])
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
            _patch_graph_activity(data, name)
            return {"success": True, "project": name}
        return {"error": f"Project '{name}' not found"}

    result = store.update_incremental(update)
    return result if result else {"error": f"Project '{name}' not found"}


def delete(category: str, index: int = None, key: str = None) -> dict:
    """Delete a memory entry by index or key with incremental graph cleanup."""
    store = get_store()

    def update(data):
        nonlocal category, index, key

        if category == "user" and key:
            if key in data["user"]:
                _release_graph_user_field(data, key, data["user"][key])
                del data["user"][key]
                return {"success": True}
            return {"error": f"Key '{key}' not found"}

        if category in ("preferences", "facts", "history", "observations"):
            if index is not None and 0 <= index < len(data[category]):
                removed = data[category].pop(index)
                if category in {"preferences", "facts", "observations"}:
                    _release_graph_entry(data, category, removed)
                return {"success": True, "removed": removed}
            return {"error": f"Invalid index {index}"}

        return {"error": f"Unknown category: {category}"}

    result = store.update_incremental(update)
    return result if result else {"error": "Update failed"}


def _memory_search_documents(data: dict) -> list[dict]:
    documents: list[dict] = []

    for key, val in data.get("user", {}).items():
        documents.append(
            {
                "payload": {"category": "user", "key": key, "value": val},
                "text": f"{key} {val}",
            }
        )

    for category in ("preferences", "facts", "history", "observations"):
        for i, entry in enumerate(data.get(category, [])):
            documents.append(
                {
                    "payload": {"category": category, "index": i, **entry},
                    "text": entry.get("content", ""),
                }
            )

    for name, act in data.get("activities", {}).items():
        documents.append(
            {
                "payload": {"category": "activity", "name": name, **act},
                "text": " ".join([name, *act.get("details", [])]),
            }
        )

    for entity in data.get("entities", {}).values():
        documents.append(
            {
                "payload": {"category": "entity", **entity},
                "text": " ".join(
                    [entity.get("name", "")]
                    + list(entity.get("aliases", []))
                    + [str(v) for v in entity.get("attrs", {}).values()]
                ),
            }
        )

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


def top_observations(n: int = 5) -> list[dict]:
    """Get top observations by mention count."""
    data = get_store().get()
    obs = data.get("observations", [])
    return sorted(obs, key=lambda x: x.get("mentions", 0), reverse=True)[:n]


def get_recent_mood(n: int = 5) -> list[dict]:
    """Get recent emotional observations."""
    data = get_store().get()
    obs = data.get("observations", [])
    sorted_obs = sorted(obs, key=lambda x: x.get("last_seen", x.get("created", "")), reverse=True)
    return sorted_obs[:n]


def get_all() -> dict:
    """Get all memory data."""
    return get_store().get()


def analyze_conversation_context(messages: list[dict]) -> dict:
    """Analyze recent conversation for lightweight interaction context."""
    if not messages:
        return {}

    user_msgs = [m["content"] for m in messages[-6:] if m["role"] == "user"]
    assistant_msgs = [m["content"] for m in messages[-6:] if m["role"] == "assistant"]

    if not user_msgs:
        return {}

    msg_count = len(user_msgs)
    total_len = sum(len(m) for m in user_msgs)
    avg_length = total_len / len(user_msgs)

    user_personal_shares = 0
    for msg in user_msgs:
        if "[MEM]" in msg:
            user_personal_shares += 1

    emotional_observations = 0
    vulnerability_markers = 0
    emotional_keywords = [
        "feel", "feeling", "frustrated", "stressed", "overwhelmed", "excited",
        "happy", "sad", "anxious", "worried", "tired", "disappointed", "nervous",
        "hope", "afraid", "scared", "proud", "accomplished", "grateful"
    ]

    for msg in user_msgs:
        emotional_observations += sum(1 for kw in emotional_keywords if kw in msg.lower())
        if any(kw in msg.lower() for kw in ["i'm worried", "i feel", "i don't know", "i'm scared", "i'm struggling", "i need help", "i'm having trouble"]):
            vulnerability_markers += 1

    asked_about_akane = False
    for msg in user_msgs:
        if any(phrase in msg.lower() for phrase in [
            "what do you think", "how do you feel", "your opinion", "do you think",
            "what's your take", "how about you", "and you", "what about you",
            "do you have thoughts", "your perspective"
        ]):
            asked_about_akane = True
            break

    reciprocal_sharing = False
    for msg in assistant_msgs[-2:]:
        if any(phrase in msg.lower() for phrase in [
            "i think", "i feel", "in my experience", "i've noticed", "i believe",
            "i'm curious", "i wonder", "i hope", "i worry", "i like", "i prefer"
        ]):
            reciprocal_sharing = True
            break

    return {
        "message_count": msg_count,
        "avg_length": avg_length,
        "user_personal_shares": user_personal_shares,
        "emotional_observations": emotional_observations,
        "vulnerability_markers": vulnerability_markers,
        "asked_about_akane": asked_about_akane,
        "reciprocal_sharing": reciprocal_sharing,
    }


def format_for_prompt() -> str:
    """Format all memories as a string for the system prompt. Uses in-memory store with caching."""
    store = get_store()
    with store._flush_lock:
        store._ensure_graph_current_unlocked()
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

    preference_lines: list[str] = []
    seen_preferences: set[str] = set()
    name = str(user.get("name", "")).strip()
    if name:
        _append_unique(preference_lines, seen_preferences, "Name", name)

    for key, value in user.items():
        if key == "name":
            continue
        lowered_key = str(key).lower()
        label = str(key).replace("_", " ").title()
        if any(token in lowered_key for token in ("communication", "tone", "style")):
            _append_unique(preference_lines, seen_preferences, "Preferred communication style", value)
        elif any(token in lowered_key for token in ("game", "genre")):
            _append_unique(preference_lines, seen_preferences, "Favorite games/genres", value)
        elif any(token in lowered_key for token in ("task", "workflow", "focus")):
            _append_unique(preference_lines, seen_preferences, "Common tasks", value)
        else:
            _append_unique(preference_lines, seen_preferences, label, value)

    favorite_preferences = [
        entry.get("content", "")
        for entry in data.get("preferences", [])
        if any(token in entry.get("content", "").lower() for token in ("game", "genre"))
    ]
    if favorite_preferences:
        _append_unique(
            preference_lines,
            seen_preferences,
            "Favorite games/genres",
            "; ".join(favorite_preferences[:3]),
        )

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

    active = {
        k: v for k, v in data.get("activities", {}).items()
        if v.get("status") == "active"
    }
    if active:
        context_lines = []
        for activity_name, proj in active.items():
            details = "; ".join(proj["details"][:2]).strip()
            if details:
                context_lines.append(f"  - Recent files or projects: {activity_name} ({details})")
            else:
                context_lines.append(f"  - Recent files or projects: {activity_name}")
        sections.append("System Context:\n" + "\n".join(context_lines))

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