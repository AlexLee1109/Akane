"""Helpers for finding relevant project files from conversational requests."""

from __future__ import annotations

import time
from pathlib import Path

_FILE_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "at", "can", "check", "code", "do", "file", "files", "for", "get",
    "give", "how", "i", "im", "implementation", "improve", "improvements", "in", "into",
    "it", "look", "me", "my", "of", "on", "project", "review", "see", "show", "suggest",
    "suggestions", "tell", "the", "this", "thought", "thoughts", "to", "what", "with", "you", "your",
}
_TEXT_SEARCH_FILE_LIMIT = 250_000
_TEXT_FILE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".css", ".html", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go", ".java",
}
_CODE_FILE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs",
    ".go", ".java",
}
_SECONDARY_TEXT_SUFFIXES = {".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".css", ".html"}
_FILE_CACHE_TTL_SECONDS = 2.0
_DEPRIORITIZED_ROOT_PREFIXES = (
    "integrations/",
    "extensions/",
    ".vscode/",
    ".idea/",
    ".zed/",
    ".github/",
    ".gitlab/",
    "dist/",
    "build/",
    "coverage/",
    "storybook-static/",
)
_EXPLICIT_INFRA_KEYWORDS = (
    "vscode",
    "vs code",
    "extension",
    "extensions",
    "bridge",
    "integration",
    "integrations",
    "launch.json",
    "editor context",
    "github actions",
    "workflow",
    "workflows",
    "ci",
    "pipeline",
    ".github",
    ".gitlab",
    "settings.json",
)
_SOURCE_ROOT_HINTS = (
    "app",
    "src",
    "lib",
    "server",
    "backend",
    "frontend",
    "client",
    "web",
    "api",
    "core",
    "pkg",
    "cmd",
)
_SELF_CODE_NOUNS = {
    "memory", "popup", "prompt", "model", "server", "client", "backend",
    "frontend", "editor", "bridge",
}


def _iter_word_tokens(text: str):
    current: list[str] = []
    for ch in str(text or ""):
        if ch.isalnum() or ch == "_":
            current.append(ch)
            continue
        if current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _iter_path_like_tokens(text: str):
    allowed = {"_", "-", ".", "/"}
    current: list[str] = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in allowed:
            current.append(ch)
            continue
        if current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _folder_reference_prefixes(text: str, roots: set[str]) -> list[str]:
    tokens = list(_iter_path_like_tokens(str(text or "").lower()))
    prefixes: list[str] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if tokens[index + 1] not in {"folder", "directory", "dir"}:
            continue
        raw = token.strip().strip("/").lower()
        if not raw:
            continue
        candidate = raw + "/"
        first = raw.split("/", 1)[0]
        if raw in roots and candidate not in seen:
            seen.add(candidate)
            prefixes.append(candidate)
            continue
        if first in roots and candidate not in seen:
            seen.add(candidate)
            prefixes.append(candidate)
    return prefixes


def _is_self_code_reference(text: str) -> bool:
    words = [token.lower() for token in _iter_word_tokens(text)]
    if not words:
        return False
    for index, word in enumerate(words):
        if word == "code" and index > 0 and words[index - 1] in {"your", "own"}:
            return True
        if (
            word == "code"
            and index > 1
            and words[index - 2] in {"your", "own"}
            and words[index - 1] in _SELF_CODE_NOUNS
        ):
            return True
        if word in _SELF_CODE_NOUNS and index > 0 and words[index - 1] == "your":
            return True
    return False


class _IndexedFile:
    __slots__ = ("path", "rel", "lowered", "stem", "tokens", "text_searchable")

    def __init__(self, path: Path, project_root: Path):
        rel = str(path.relative_to(project_root))
        lowered = rel.lower()
        self.path = path
        self.rel = rel
        self.lowered = lowered
        self.stem = path.stem.lower()
        self.tokens = {part for part in _iter_word_tokens(lowered)}
        self.text_searchable = path.suffix.lower() in _TEXT_FILE_SUFFIXES


class CodebaseSearch:
    """Search the current workspace for files relevant to a natural-language request."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self._file_index: dict[str, list[str]] | None = None
        self._inventory: list[_IndexedFile] | None = None
        self._inventory_built_at = 0.0
        self._text_cache: dict[str, tuple[int, int, str]] = {}

    def _inventory_entries(self) -> list[_IndexedFile]:
        now = time.monotonic()
        if self._inventory is not None and (now - self._inventory_built_at) < _FILE_CACHE_TTL_SECONDS:
            return self._inventory

        inventory: list[_IndexedFile] = []
        index: dict[str, list[str]] = {}
        for path in self.project_root.rglob("*"):
            if not path.is_file() or self._should_skip_path(path):
                continue
            entry = _IndexedFile(path, self.project_root)
            inventory.append(entry)
            index.setdefault(path.name.lower(), []).append(entry.rel)

        self._inventory = inventory
        self._file_index = index
        self._inventory_built_at = now
        return inventory

    def _root_directory_names(self) -> set[str]:
        roots: set[str] = set()
        for entry in self._inventory_entries():
            first = entry.rel.split("/", 1)[0].strip().lower()
            if first:
                roots.add(first)
        return roots

    def _primary_source_prefixes(self) -> list[str]:
        counts: dict[str, int] = {}
        for entry in self._inventory_entries():
            rel = entry.rel.strip().lower()
            if not rel or any(rel.startswith(prefix) for prefix in _DEPRIORITIZED_ROOT_PREFIXES):
                continue
            first = rel.split("/", 1)[0]
            if entry.path.suffix.lower() in _CODE_FILE_SUFFIXES:
                counts[first] = counts.get(first, 0) + 1

        if not counts:
            return []

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        preferred: list[str] = []
        seen: set[str] = set()
        for hint in _SOURCE_ROOT_HINTS:
            if hint in counts and hint not in seen:
                seen.add(hint)
                preferred.append(hint + "/")
        for name, count in ranked:
            if count < 2:
                continue
            if name not in seen:
                seen.add(name)
                preferred.append(name + "/")
            if len(preferred) >= 4:
                break
        return preferred

    def _explicit_folder_prefixes(self, user_input: str) -> list[str]:
        roots = self._root_directory_names()
        return _folder_reference_prefixes(user_input, roots)

    def _read_lower_text(self, entry: _IndexedFile) -> str:
        try:
            stat = entry.path.stat()
        except OSError:
            return ""

        size = int(stat.st_size)
        if size > _TEXT_SEARCH_FILE_LIMIT:
            return ""

        cache_key = entry.rel
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        cached = self._text_cache.get(cache_key)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]

        try:
            text = entry.path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return ""

        self._text_cache[cache_key] = (mtime_ns, size, text)
        return text

    @staticmethod
    def _normalize_values(values: list[str] | None) -> list[str]:
        return [str(value or "").strip() for value in (values or []) if str(value or "").strip()]

    def _is_explicit_infra_context(self, lowered_input: str) -> bool:
        return any(token in lowered_input for token in _EXPLICIT_INFRA_KEYWORDS)

    def _candidate_search_context(self, user_input: str) -> dict[str, object]:
        lowered = str(user_input or "").lower()
        focused_prefixes = self._explicit_folder_prefixes(user_input)
        self_code_context = _is_self_code_reference(lowered)
        preferred_prefixes = (
            []
            if focused_prefixes
            else (self._primary_source_prefixes() if self_code_context else [])
        )
        return {
            "lowered": lowered,
            "focused_prefixes": focused_prefixes,
            "preferred_prefixes": preferred_prefixes,
            "integration_context": self._is_explicit_infra_context(lowered),
            "self_code_context": self_code_context,
        }

    def _path_allowed(self, rel: str, *, focused_prefixes: list[str], integration_context: bool) -> bool:
        path = str(rel or "").strip().lower()
        if not path:
            return False
        if focused_prefixes:
            return any(path.startswith(prefix) for prefix in focused_prefixes)
        if integration_context:
            return True
        if any(path.startswith(prefix) for prefix in _DEPRIORITIZED_ROOT_PREFIXES):
            return False
        if any(
            marker in path
            for marker in (
                "/.vscode/",
                "/.idea/",
                "/.zed/",
                "/.github/",
                "/.gitlab/",
            )
        ):
            return False
        return True

    @staticmethod
    def _prioritize_paths(paths: list[str], preferred_prefixes: list[str]) -> list[str]:
        if not preferred_prefixes:
            return list(paths)
        preferred = [rel for rel in paths if any(rel.lower().startswith(prefix) for prefix in preferred_prefixes)]
        other = [rel for rel in paths if rel not in preferred]
        return preferred + other

    @staticmethod
    def _append_unique(candidates: list[str], seen: set[str], rel: str) -> bool:
        if rel in seen:
            return False
        seen.add(rel)
        candidates.append(rel)
        return True

    def get_project_file_index(self) -> dict[str, list[str]]:
        if self._file_index is not None:
            self._inventory_entries()
            return self._file_index or {}

        self._inventory_entries()
        return self._file_index or {}

    def normalize_search_token(self, token: str) -> str:
        value = token.lower().strip("_-. ")
        if len(value) > 4 and value.endswith("ies"):
            value = value[:-3] + "y"
        elif len(value) > 3 and value.endswith("es"):
            value = value[:-2]
        elif len(value) > 3 and value.endswith("s"):
            value = value[:-1]
        for suffix in ("ing", "ed"):
            if len(value) > len(suffix) + 2 and value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        return value

    def extract_query_tokens(self, user_input: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for raw in _iter_word_tokens(str(user_input or "").lower()):
            token = self.normalize_search_token(raw)
            if len(token) <= 1 or token in _FILE_SEARCH_STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def has_recent_topic_overlap(self, user_input: str, recent_text: str) -> bool:
        current_tokens = set(self.extract_query_tokens(user_input))
        recent_tokens = set(self.extract_query_tokens(recent_text))
        if not current_tokens or not recent_tokens:
            return False
        return bool(current_tokens & recent_tokens)

    def looks_like_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in _TEXT_FILE_SUFFIXES

    def search_paths(self, user_input: str, limit: int = 3) -> list[str]:
        query_tokens = self.extract_query_tokens(user_input)
        if not query_tokens:
            return []

        scored: list[tuple[int, str]] = []
        for entry in self._inventory_entries():
            score = self._path_score(entry, query_tokens)
            if score > 0:
                scored.append((score, entry.rel))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in scored[:limit]]

    def search_contents(self, user_input: str, limit: int = 3) -> list[str]:
        query_tokens = self.extract_query_tokens(user_input)
        if not query_tokens:
            return []

        scored: list[tuple[int, str]] = []
        for entry in self._inventory_entries():
            if not entry.text_searchable:
                continue
            text = self._read_lower_text(entry)
            if not text:
                continue

            score = 0
            for token in query_tokens:
                hits = text.count(token)
                if hits:
                    score += min(hits, 6)
                    if f"def _{token}" in text or f"{token}(" in text:
                        score += 3
            if score > 0:
                scored.append((score, entry.rel))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in scored[:limit]]

    def candidate_paths(
        self,
        user_input: str,
        *,
        file_refs: list[str] | None = None,
        file_ref_pattern=None,
        recent_targets: list[str] | None = None,
        active_file: str = "",
        open_tabs: list[str] | None = None,
        recent_texts: list[str] | None = None,
        coding_request: bool = False,
        explicit_file_reference: bool = False,
        followup_reference: bool = False,
        reuse_recent_targets: bool = False,
        limit: int = 3,
    ) -> list[str]:
        recent_targets = self._normalize_values(recent_targets)
        open_tabs = self._normalize_values(open_tabs)
        recent_texts = self._normalize_values(recent_texts)
        normalized_file_refs = self._normalize_values(file_refs)

        if not normalized_file_refs and file_ref_pattern is not None:
            extracted_refs: list[str] = []
            seen_refs: set[str] = set()
            try:
                matches = file_ref_pattern.finditer(str(user_input or ""))
            except AttributeError:
                matches = ()
            for match in matches:
                try:
                    raw = str(match.group(1) or "").strip()
                except IndexError:
                    continue
                if raw and raw not in seen_refs:
                    seen_refs.add(raw)
                    extracted_refs.append(raw)
            normalized_file_refs = extracted_refs

        seen: set[str] = set()
        candidates: list[str] = []
        context = self._candidate_search_context(user_input)
        lowered = str(context["lowered"])
        focused_prefixes = list(context["focused_prefixes"])
        preferred_prefixes = list(context["preferred_prefixes"])
        integration_context = bool(context["integration_context"])
        self_code_context = bool(context.get("self_code_context"))

        for raw_value in normalized_file_refs:
            raw = str(raw_value or "").strip("`'\".,:;()[]{}")
            if not raw:
                continue
            if "/" in raw:
                resolved = (self.project_root / raw).resolve()
                if resolved.is_file() and str(resolved).startswith(str(self.project_root)):
                    rel = str(resolved.relative_to(self.project_root))
                    if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                        self._append_unique(candidates, seen, rel)
                continue

            matches = self.get_project_file_index().get(raw.lower(), [])
            if len(matches) == 1 and self._path_allowed(
                matches[0],
                focused_prefixes=focused_prefixes,
                integration_context=integration_context,
            ):
                self._append_unique(candidates, seen, matches[0])

        if (
            active_file
            and self._path_allowed(active_file, focused_prefixes=focused_prefixes, integration_context=integration_context)
            and any(phrase in lowered for phrase in {"this file", "current file", "that file", "the file"})
            and active_file not in seen
        ):
            seen.add(active_file)
            candidates.insert(0, active_file)

        query_tokens = self.extract_query_tokens(user_input)
        search_queries = [user_input]

        direct_query_scored = self._score_paths(user_input, limit=max(6, limit))
        has_strong_direct_match = bool(direct_query_scored and direct_query_scored[0][0] >= 8)

        if recent_targets and reuse_recent_targets and coding_request and not explicit_file_reference:
            for rel in recent_targets:
                if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                    self._append_unique(candidates, seen, rel)

        if (recent_targets or followup_reference or len(query_tokens) <= 3) and not (
            self_code_context and has_strong_direct_match and not reuse_recent_targets
        ):
            search_queries.extend(recent_texts)

        for remembered in recent_targets:
            if followup_reference and self._path_allowed(
                remembered,
                focused_prefixes=focused_prefixes,
                integration_context=integration_context,
            ):
                self._append_unique(candidates, seen, remembered)

        strong_path_hits = 0
        for query in search_queries:
            prior_count = len(candidates)
            query_scored = direct_query_scored if query == user_input else self._score_paths(query, limit=6)
            query_scored.sort(
                key=lambda item: (
                    0 if any(item[1].lower().startswith(prefix) for prefix in preferred_prefixes) else 1,
                    -item[0],
                    item[1],
                )
            )
            for score, rel in query_scored:
                if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                    self._append_unique(candidates, seen, rel)
                if score >= 7:
                    strong_path_hits += 1
            if strong_path_hits < max(1, limit) and (len(candidates) < limit or len(candidates) == prior_count):
                for rel in self._prioritize_paths(self.search_contents(query, limit=max(2, limit)), preferred_prefixes):
                    if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                        self._append_unique(candidates, seen, rel)

        for rel in self._prioritize_paths(recent_targets, preferred_prefixes):
            if not reuse_recent_targets:
                continue
            if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                self._append_unique(candidates, seen, rel)

        for rel in self._prioritize_paths(open_tabs[:5], preferred_prefixes):
            if len(candidates) >= limit:
                break
            if self._path_allowed(rel, focused_prefixes=focused_prefixes, integration_context=integration_context):
                self._append_unique(candidates, seen, rel)

        return candidates[:limit]

    def _score_paths(self, user_input: str, limit: int = 3) -> list[tuple[int, str]]:
        query_tokens = self.extract_query_tokens(user_input)
        if not query_tokens:
            return []

        scored: list[tuple[int, str]] = []
        for entry in self._inventory_entries():
            score = self._path_score(entry, query_tokens)
            if score > 0:
                scored.append((score, entry.rel))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:limit]

    def _path_score(self, entry: _IndexedFile, query_tokens: list[str]) -> int:
        path_tokens = {self.normalize_search_token(part) for part in entry.tokens}
        score = 0
        exact_hits = 0
        for token in query_tokens:
            if entry.stem == token:
                score += 10
                exact_hits += 1
            elif self.normalize_search_token(entry.stem) == token:
                score += 8
                exact_hits += 1
            elif token in path_tokens:
                score += 6
                exact_hits += 1
            elif token in entry.stem:
                score += 4
            elif token in entry.lowered:
                score += 2
        if exact_hits >= 2:
            score += 4
        suffix = entry.path.suffix.lower()
        if suffix in _CODE_FILE_SUFFIXES:
            score += 1
        elif suffix in _SECONDARY_TEXT_SUFFIXES:
            score -= 1
        return score

    @staticmethod
    def _should_skip_path(path: Path) -> bool:
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
        return any(part in skip_dirs for part in path.parts)
