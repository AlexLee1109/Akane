"""Helpers for finding relevant project files from conversational requests."""

from __future__ import annotations

import os
import time
from functools import lru_cache
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
_BINARY_FILE_SUFFIXES = {
    ".node", ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".obj",
    ".class", ".jar", ".war", ".pyd", ".whl", ".zip", ".tar", ".gz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
}
_DEPRIORITIZED_ROOT_PREFIXES = (
    "integrations/", "extensions/", ".vscode-server/", ".vscode/", ".idea/", ".zed/",
    ".github/", ".gitlab/", "dist/", "build/", "coverage/", "storybook-static/",
)
_REFERENCE_TRAILING_PUNCT = " \t\r\n`'\".,:;()[]{}"
_FILE_CACHE_TTL_SECONDS = 5.0
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    ".vscode-server", ".mypy_cache", ".pytest_cache", ".cache", "target", "out",
}
_PROJECT_PRIORITY_PREFIXES = ("app/", "src/", "lib/", "core/", "server/", "api/", "ui/", "web/")


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


def _clean_reference_path(raw_ref: str) -> str:
    raw = str(raw_ref or "").strip(_REFERENCE_TRAILING_PUNCT).replace("\\", "/")
    if not raw or "://" in raw or raw.startswith("~"):
        return ""
    if raw.startswith("./"):
        raw = raw[2:]
    if ":" in raw:
        path_part, _, suffix = raw.partition(":")
        suffix = suffix.strip()
        if path_part and suffix and (suffix[0].isdigit() or suffix.lower().startswith("l")):
            raw = path_part.strip()
        else:
            return ""
    return raw


class _IndexedFile:
    __slots__ = ("path", "rel", "lowered", "stem", "normalized_stem", "tokens", "normalized_tokens", "text_searchable")

    def __init__(self, path: Path, project_root: Path):
        rel = str(path.relative_to(project_root))
        lowered = rel.lower()
        self.path = path
        self.rel = rel
        self.lowered = lowered
        self.stem = path.stem.lower()
        self.normalized_stem = CodebaseSearch.normalize_search_token_static(self.stem)
        self.tokens = {part for part in _iter_word_tokens(lowered)}
        self.normalized_tokens = {CodebaseSearch.normalize_search_token_static(p) for p in self.tokens}
        self.text_searchable = path.suffix.lower() in _TEXT_FILE_SUFFIXES


class CodebaseSearch:
    """Search the current workspace for files relevant to a natural-language request."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self._file_index: dict[str, list[str]] | None = None
        self._inventory: list[_IndexedFile] | None = None
        self._inventory_built_at = 0.0
        self._text_cache: dict[str, tuple[int, int, str]] = {}

    # ------------------------------------------------------------------ #
    # Inventory                                                            #
    # ------------------------------------------------------------------ #

    def _inventory_entries(self) -> list[_IndexedFile]:
        now = time.monotonic()
        if self._inventory is not None and (now - self._inventory_built_at) < _FILE_CACHE_TTL_SECONDS:
            return self._inventory
        inventory: list[_IndexedFile] = []
        index: dict[str, list[str]] = {}
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
            root_path = Path(root)
            for filename in files:
                path = root_path / filename
                if self._skip_file(path):
                    continue
                entry = _IndexedFile(path, self.project_root)
                inventory.append(entry)
                index.setdefault(filename.lower(), []).append(entry.rel)
                index.setdefault(path.stem.lower(), []).append(entry.rel)
        self._inventory = inventory
        self._file_index = index
        self._inventory_built_at = now
        return inventory

    def get_project_file_index(self) -> dict[str, list[str]]:
        self._inventory_entries()
        return self._file_index or {}

    # ------------------------------------------------------------------ #
    # Token normalisation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    @lru_cache(maxsize=4096)
    def normalize_search_token_static(token: str) -> str:
        value = str(token or "").lower().strip("_-. ")
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

    def normalize_search_token(self, token: str) -> str:
        return self.normalize_search_token_static(token)

    @lru_cache(maxsize=1024)
    def _cached_query_tokens(self, user_input: str) -> tuple[str, ...]:
        tokens: list[str] = []
        seen: set[str] = set()
        for raw in _iter_word_tokens(str(user_input or "").lower()):
            token = self.normalize_search_token_static(raw)
            if len(token) <= 1 or token in _FILE_SEARCH_STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tuple(tokens)

    def extract_query_tokens(self, user_input: str) -> list[str]:
        return list(self._cached_query_tokens(str(user_input or "")))

    @staticmethod
    def _skip_file(path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in _BINARY_FILE_SUFFIXES:
            return True
        try:
            return path.stat().st_size > _TEXT_SEARCH_FILE_LIMIT and suffix not in _CODE_FILE_SUFFIXES
        except OSError:
            return True

    def is_source_candidate(self, rel: str) -> bool:
        rel_path = self._project_relative_path(rel)
        lowered = rel_path.lower()
        if not lowered:
            return False
        suffix = Path(lowered).suffix
        if suffix in _BINARY_FILE_SUFFIXES:
            return False
        return not any(lowered.startswith(prefix) for prefix in _DEPRIORITIZED_ROOT_PREFIXES)

    def _project_relative_path(self, raw_path: str, *, must_be_file: bool = False) -> str:
        raw = _clean_reference_path(raw_path)
        if not raw:
            return ""
        try:
            candidate = Path(raw)
            resolved = candidate.resolve() if candidate.is_absolute() else (self.project_root / candidate).resolve()
            rel = resolved.relative_to(self.project_root)
        except (OSError, ValueError):
            return ""
        if must_be_file and not resolved.is_file():
            return ""
        return str(rel)

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #

    def _path_score(self, entry: _IndexedFile, query_tokens: list[str]) -> int:
        suffix = entry.path.suffix.lower()
        if suffix in _BINARY_FILE_SUFFIXES:
            return 0
        score = 0
        exact_hits = 0
        path_parts = entry.lowered.split("/")
        for token in query_tokens:
            if entry.stem == token:
                score += 10; exact_hits += 1
            elif entry.normalized_stem == token:
                score += 8; exact_hits += 1
            elif token in entry.normalized_tokens:
                score += 6; exact_hits += 1
            elif any(part.startswith(token) for part in path_parts):
                score += 5
            elif token in entry.stem:
                score += 4
            elif token in entry.lowered:
                score += 2
        if exact_hits >= 2:
            score += 4
        if entry.lowered.startswith(_PROJECT_PRIORITY_PREFIXES):
            score += 2
        if any(entry.lowered.startswith(p) for p in _DEPRIORITIZED_ROOT_PREFIXES):
            score -= 6
        if suffix in _CODE_FILE_SUFFIXES:
            score += 2
        elif suffix in _SECONDARY_TEXT_SUFFIXES:
            score -= 1
        return score

    def _score_all(self, user_input: str, *, limit: int = 8) -> list[tuple[int, str]]:
        tokens = self.extract_query_tokens(user_input)
        if not tokens:
            return []
        scored = [
            (s, e.rel)
            for e in self._inventory_entries()
            if (s := self._path_score(e, tokens)) > 0
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:limit]

    # ------------------------------------------------------------------ #
    # Content search                                                       #
    # ------------------------------------------------------------------ #

    def _read_lower_text(self, entry: _IndexedFile) -> str:
        try:
            stat = entry.path.stat()
        except OSError:
            return ""
        size = int(stat.st_size)
        if size > _TEXT_SEARCH_FILE_LIMIT:
            return ""
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        cached = self._text_cache.get(entry.rel)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
        try:
            text = entry.path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return ""
        self._text_cache[entry.rel] = (mtime_ns, size, text)
        return text

    def search_contents(self, user_input: str, limit: int = 3) -> list[str]:
        tokens = self.extract_query_tokens(user_input)
        if not tokens:
            return []
        scored: list[tuple[int, str]] = []
        for entry in self._inventory_entries():
            if not entry.text_searchable:
                continue
            text = self._read_lower_text(entry)
            if not text:
                continue
            present = [t for t in tokens if t in text]
            if not present:
                continue
            score = sum(
                min(text.count(t), 6) + (3 if f"def _{t}" in text or f"{t}(" in text else 0)
                for t in present
            )
            if score > 0:
                scored.append((score, entry.rel))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [r for _, r in scored[:limit]]

    # ------------------------------------------------------------------ #
    # Reference resolution                                                 #
    # ------------------------------------------------------------------ #

    def _resolve_ref(self, raw_ref: str) -> list[str]:
        """Resolve a file-reference token to zero or more repo-relative paths."""
        raw = _clean_reference_path(raw_ref)
        if not raw:
            return []
        # Exact path on disk
        if "/" in raw or Path(raw).is_absolute():
            rel = self._project_relative_path(raw, must_be_file=True)
            if rel:
                return [rel]
        index = self.get_project_file_index()
        lowered = raw.lower()
        bare = lowered.rsplit("/", 1)[-1]
        stem = bare.rsplit(".", 1)[0]
        # Exact basename / stem lookup
        matches = list(dict.fromkeys(index.get(bare, []) + index.get(stem, [])))
        if matches:
            return matches[:3]
        # Fuzzy: substring or token-subset match
        raw_tokens = set(self.extract_query_tokens(raw))
        fuzzy = [
            e.rel for e in self._inventory_entries()
            if (bare and (e.lowered.endswith("/" + bare) or e.lowered == bare))
            or (stem and stem == e.stem)
            or (raw_tokens and raw_tokens.issubset(e.normalized_tokens))
        ]
        return fuzzy[:3]

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def candidate_paths(
        self,
        user_input: str,
        *,
        file_refs: list[str] | None = None,
        file_ref_pattern=None,           # accepted but unused (refs are passed pre-extracted)
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
        file_refs     = [s for s in (file_refs     or []) if s and s.strip()]
        recent_targets = [p for s in (recent_targets or []) if (p := self._project_relative_path(s, must_be_file=True))]
        open_tabs     = [p for s in (open_tabs     or []) if (p := self._project_relative_path(s, must_be_file=True))]
        recent_texts  = [s for s in (recent_texts  or []) if s and s.strip()]

        seen: set[str] = set()
        candidates: list[str] = []

        def _add(rel: str) -> bool:
            if rel and rel not in seen:
                seen.add(rel)
                candidates.append(rel)
                return True
            return False

        # 1. Explicit file references (highest priority)
        for ref in file_refs:
            for rel in self._resolve_ref(ref):
                _add(rel)

        # 2. Active file when the message refers to it by phrase
        lowered_input = str(user_input or "").lower()
        active_file = self._project_relative_path(active_file, must_be_file=True)
        if active_file and active_file not in seen and any(
            phrase in lowered_input for phrase in ("this file", "current file", "that file", "the file")
        ):
            _add(active_file)

        # 3. Carry recent confirmed targets for followups before fuzzy search.
        if reuse_recent_targets or followup_reference:
            for rel in recent_targets:
                if self.is_source_candidate(rel):
                    _add(rel)

        # 4. Score-based path search on the current query
        for _score, rel in self._score_all(user_input, limit=max(10, limit * 3)):
            if self.is_source_candidate(rel) or explicit_file_reference:
                _add(rel)

        # 5. Content search if still short
        if len(candidates) < limit:
            for rel in self.search_contents(user_input, limit=limit):
                if self.is_source_candidate(rel):
                    _add(rel)

        # 6. Score recent context text for additional signal
        if len(candidates) < limit and recent_texts:
            combined = " ".join(recent_texts[:3])
            for _score, rel in self._score_all(combined, limit=limit):
                if self.is_source_candidate(rel):
                    _add(rel)

        # 7. Open tabs as last resort
        for rel in open_tabs[:5]:
            if len(candidates) >= limit:
                break
            if self.is_source_candidate(rel):
                _add(rel)

        return candidates[:limit]
