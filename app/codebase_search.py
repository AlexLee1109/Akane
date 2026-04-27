"""Helpers for finding relevant project files from conversational requests."""

from __future__ import annotations

import re
import time
from pathlib import Path

_FILE_SEARCH_STOPWORDS = {
    "a", "an", "and", "at", "can", "check", "code", "do", "file", "files", "for", "get",
    "give", "how", "i", "im", "implementation", "improve", "improvements", "in", "into",
    "it", "look", "me", "my", "of", "on", "project", "review", "see", "show", "suggest",
    "suggestions", "tell", "the", "this", "to", "what", "with", "you",
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
_QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_FILE_CACHE_TTL_SECONDS = 2.0


class _IndexedFile:
    __slots__ = ("path", "rel", "lowered", "stem", "tokens", "text_searchable")

    def __init__(self, path: Path, project_root: Path):
        rel = str(path.relative_to(project_root))
        lowered = rel.lower()
        self.path = path
        self.rel = rel
        self.lowered = lowered
        self.stem = path.stem.lower()
        self.tokens = {part for part in _QUERY_TOKEN_PATTERN.findall(lowered)}
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
        for raw in _QUERY_TOKEN_PATTERN.findall(str(user_input or "").lower()):
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
        file_ref_pattern: re.Pattern[str],
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
        recent_targets = [str(path or "").strip() for path in (recent_targets or []) if str(path or "").strip()]
        open_tabs = [str(path or "").strip() for path in (open_tabs or []) if str(path or "").strip()]
        recent_texts = [str(text or "").strip() for text in (recent_texts or []) if str(text or "").strip()]

        seen: set[str] = set()
        candidates: list[str] = []
        lowered = str(user_input or "").lower()

        for match in file_ref_pattern.finditer(str(user_input or "")):
            raw = match.group(1).strip("`'\".,:;()[]{}")
            if not raw:
                continue
            if "/" in raw:
                resolved = (self.project_root / raw).resolve()
                if resolved.is_file() and str(resolved).startswith(str(self.project_root)):
                    rel = str(resolved.relative_to(self.project_root))
                    if rel not in seen:
                        seen.add(rel)
                        candidates.append(rel)
                continue

            matches = self.get_project_file_index().get(raw.lower(), [])
            if len(matches) == 1 and matches[0] not in seen:
                seen.add(matches[0])
                candidates.append(matches[0])

        if active_file and any(phrase in lowered for phrase in {"this file", "current file", "that file", "the file"}):
            if active_file not in seen:
                seen.add(active_file)
                candidates.insert(0, active_file)

        query_tokens = self.extract_query_tokens(user_input)
        search_queries = [user_input]
        if recent_targets and reuse_recent_targets and coding_request and not explicit_file_reference:
            for rel in recent_targets:
                if rel not in seen:
                    seen.add(rel)
                    candidates.append(rel)

        if recent_targets or followup_reference or len(query_tokens) <= 3:
            search_queries.extend(recent_texts)

        for remembered in recent_targets:
            if followup_reference and remembered not in seen:
                seen.add(remembered)
                candidates.append(remembered)

        strong_path_hits = 0
        for query in search_queries:
            prior_count = len(candidates)
            query_scored = self._score_paths(query, limit=6)
            for score, rel in query_scored:
                if rel not in seen:
                    seen.add(rel)
                    candidates.append(rel)
                if score >= 7:
                    strong_path_hits += 1
            if strong_path_hits < max(1, limit) and (len(candidates) < limit or len(candidates) == prior_count):
                for rel in self.search_contents(query, limit=max(2, limit)):
                    if rel not in seen:
                        seen.add(rel)
                        candidates.append(rel)

        for rel in recent_targets:
            if not reuse_recent_targets:
                continue
            if rel not in seen:
                seen.add(rel)
                candidates.append(rel)

        for rel in open_tabs[:5]:
            if len(candidates) >= limit:
                break
            if rel and rel not in seen:
                seen.add(rel)
                candidates.append(rel)

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
