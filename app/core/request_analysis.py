"""Small compatibility request analyzer."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.agents.codebase_search import CodebaseSearch

_FILE_SUFFIXES = (".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".sh")
_CODE_WORDS = {
    "code", "file", "files", "bug", "error", "fix", "edit", "rewrite", "refactor",
    "function", "class", "server", "python", "javascript", "typescript", "vscode",
}
_FOLLOWUP_WORDS = {"it", "this", "that", "those", "them", "same", "still", "again"}
_DETAIL_WORDS = {"detail", "details", "explain", "walkthrough", "deep", "full"}
_SUGGEST_WORDS = {"suggest", "suggestion", "improve", "recommend"}
_SMALLTALK = {"hi", "hello", "hey", "thanks", "thank", "bye", "cool", "nice"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip(".,:;()[]{}'\"`") for m in _TOKEN_RE.finditer(str(text or ""))]


def _pathish(token: str) -> bool:
    return "/" in token or token.endswith(_FILE_SUFFIXES)


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    last_user: str = ""
    last_assistant: str = ""
    recent_code_targets: tuple[str, ...] = ()
    active_file: str = ""
    open_tabs: tuple[str, ...] = ()
    recent_text: str = ""
    editor_connected: bool = False


@dataclass(frozen=True, slots=True)
class RequestAnalysis:
    query_tokens: tuple[str, ...]
    open_vscode: bool
    coding: bool
    codebase_direct: bool
    codebase_followup: bool
    explicit_file_reference: bool
    referential_followup: bool
    followup_reference: bool
    explanation_followup: bool
    affirmative_followup: bool
    assistant_invited_continuation: bool
    topic_overlap: bool
    topic_shift: bool
    wants_execution: bool
    should_carry_last_reply: bool
    wants_detail: bool
    wants_single_suggestion: bool
    wants_brainstorm: bool
    self_identity_request: bool
    smalltalk: bool
    should_force_vscode: bool

    @property
    def codebase_context(self) -> bool:
        return self.codebase_direct or self.codebase_followup

    @property
    def coding_like(self) -> bool:
        return self.coding or self.codebase_context


class RequestAnalyzer:
    def __init__(self, search: CodebaseSearch):
        self.search = search

    def extract_file_refs(self, text: str) -> list[str]:
        seen: set[str] = set()
        refs: list[str] = []
        for token in _tokens(text):
            if not _pathish(token):
                continue
            value = token.strip("`'\".,:;()[]{}")
            if value and value not in seen:
                seen.add(value)
                refs.append(value)
        return refs

    def analyze(self, user_input: str, snapshot: RequestSnapshot) -> RequestAnalysis:
        toks = _tokens(user_input)
        token_set = set(toks)
        lowered = " ".join(toks)
        query_tokens = tuple(self.search.extract_query_tokens(user_input))
        explicit_file_reference = bool(self.extract_file_refs(user_input))
        has_code_word = bool(token_set & _CODE_WORDS or any(_pathish(t) for t in toks))
        has_code_context = bool(
            snapshot.recent_code_targets
            or snapshot.active_file
            or any(_pathish(t) or t in _CODE_WORDS for t in _tokens(snapshot.recent_text + " " + snapshot.last_user + " " + snapshot.last_assistant))
        )
        followup_reference = bool(token_set & _FOLLOWUP_WORDS)
        referential_followup = bool(token_set & {"it", "this", "that", "those", "them", "same"})
        affirmative_followup = lowered in {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "do it"}
        explanation_followup = bool(token_set & {"explain", "more", "why", "how"})
        open_vscode = bool(toks[:1] and toks[0] in {"open", "launch", "start"} and ("vscode" in token_set or "code" in token_set))
        wants_execution = bool(token_set & {"do", "fix", "edit", "change", "apply", "rewrite", "update", "make"})
        wants_detail = bool(token_set & _DETAIL_WORDS)
        wants_single_suggestion = bool(token_set & _SUGGEST_WORDS and not wants_detail)
        self_identity_request = bool(toks[:1] in (["who"], ["what"]) and ("you" in token_set or "your" in token_set))
        smalltalk = bool(token_set and token_set <= _SMALLTALK)
        topic_shift = bool(toks[:1] and toks[0] in {"also", "instead", "unrelated", "separately", "different"})
        topic_overlap = bool(set(query_tokens) & set(self.search.extract_query_tokens(snapshot.recent_text or "")))
        codebase_direct = bool(explicit_file_reference or (has_code_word and not smalltalk))
        codebase_followup = bool(has_code_context and not topic_shift and (followup_reference or affirmative_followup or wants_execution or explanation_followup or topic_overlap))
        coding = bool(codebase_direct or (has_code_word and not smalltalk))
        should_carry = bool(
            (snapshot.last_user or snapshot.last_assistant or has_code_context)
            and not topic_shift
            and (followup_reference or affirmative_followup or explanation_followup or codebase_followup or topic_overlap)
        )
        assistant_invited = "?" in str(snapshot.last_assistant or "") and any(
            phrase in str(snapshot.last_assistant).lower()
            for phrase in ("want me", "should i", "i can", "would you like")
        )
        wants_brainstorm = bool("?" in str(user_input) and token_set & {"idea", "ideas", "feature", "build", "make"})
        should_force_vscode = bool(open_vscode or (snapshot.editor_connected and wants_execution and coding))

        if self_identity_request or smalltalk:
            coding = codebase_direct = codebase_followup = wants_execution = should_force_vscode = False

        return RequestAnalysis(
            query_tokens=query_tokens,
            open_vscode=open_vscode,
            coding=coding,
            codebase_direct=codebase_direct,
            codebase_followup=codebase_followup,
            explicit_file_reference=explicit_file_reference,
            referential_followup=referential_followup,
            followup_reference=followup_reference,
            explanation_followup=explanation_followup,
            affirmative_followup=affirmative_followup,
            assistant_invited_continuation=assistant_invited,
            topic_overlap=topic_overlap,
            topic_shift=topic_shift,
            wants_execution=wants_execution,
            should_carry_last_reply=should_carry,
            wants_detail=wants_detail,
            wants_single_suggestion=wants_single_suggestion,
            wants_brainstorm=wants_brainstorm,
            self_identity_request=self_identity_request,
            smalltalk=smalltalk,
            should_force_vscode=should_force_vscode,
        )

    def candidate_paths(self, user_input: str, snapshot: RequestSnapshot, *, limit: int = 3) -> list[str]:
        analysis = self.analyze(user_input, snapshot)
        return self.search.candidate_paths(
            user_input,
            file_refs=self.extract_file_refs(user_input),
            recent_targets=list(snapshot.recent_code_targets),
            active_file=snapshot.active_file,
            open_tabs=list(snapshot.open_tabs),
            recent_texts=[snapshot.recent_text, snapshot.last_user, snapshot.last_assistant],
            coding_request=analysis.coding,
            explicit_file_reference=analysis.explicit_file_reference,
            followup_reference=analysis.followup_reference,
            reuse_recent_targets=analysis.should_carry_last_reply,
            limit=limit,
        )
