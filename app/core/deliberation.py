"""Structured assessment of working context; never selects or performs actions."""

import hashlib
import json
from dataclasses import asdict, dataclass

from app.core.attention import _terms


ASSESSMENT_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class DeliberationAssessment:
    profile_id: str
    percept_id: str
    frame_reference: str
    mode: str
    complexity: str
    confidence: str
    uncertainty_codes: tuple[str, ...]
    conflict_codes: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    verification_needed: bool
    reason_codes: tuple[str, ...]
    created_at: float

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate observation field.')
        result[key] = value
    return result


def _state_report(percept):
    """Optional exact data shape in a local receipt, not a control protocol.

    No free-text assertion extraction. Subject must agree with the receipt's
    provenance envelope. Extra fields have no effect; duplicate keys fail closed.
    """
    if percept.source not in {'runtime', 'tool'}:
        return None
    try:
        row = json.loads(percept.content, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(row, dict) or not {'entity_id', 'attribute', 'value'} <= row.keys()
            or any(not isinstance(row[k], str) or not row[k] or row[k] != row[k].strip()
                   for k in ('entity_id', 'attribute', 'value'))
            or row['entity_id'] != percept.subject
            or row['attribute'] != row['attribute'].casefold()):
        return None
    return row


def _assess_frame(frame):
    """Internal classifier. Store reconstructs a current frame before calling."""
    current = []
    history = []
    relations = []
    for item in frame.relevant_world:
        row = dict(item.values)
        if item.kind == 'world_history':
            history.append(item)
        elif item.kind == 'world_current':
            if {'entity_id', 'attribute', 'value'} <= row.keys():
                current.append((item, row))
            elif 'relation' in row:
                relations.append(item)

    conflicts, missing, uncertainty, reasons = [], [], [], []
    values_by_key = {}
    for item, row in current:
        if item.sources:
            key = (row['entity_id'], row['attribute'])
            values_by_key.setdefault(key, set()).add(row['value'])
    if any(len(values) > 1 for values in values_by_key.values()):
        conflicts.append('incompatible_current_state')

    report = _state_report(frame.percept)
    corroborated = False
    if report:
        matches = [(item, row) for item, row in current
                   if (row['entity_id'], row['attribute']) == (report['entity_id'], report['attribute'])]
        if not matches:
            uncertainty.append('observation_unconfirmed')
        elif any(frame.percept.timestamp < item.timestamp for item, _ in matches):
            uncertainty.append('observation_older_than_state')
        elif any(row['value'] != report['value'] for _, row in matches):
            conflicts.append('observation_world')
        else:
            corroborated = True

    # A failure word is only an observed problem cue. It grants no authority;
    # both an existing objective and missing structured evidence are required.
    problem = bool(frame.objective and (
        report or _terms(frame.percept.content) & {'failed', 'failure', 'broke', 'broken', 'error'}
    ))
    outcomes = [r for r in frame.linked_evidence if r.kind == 'outcome' and r.sources]
    required_current = current
    if report:
        required_current = [(item, row) for item, row in current
                            if (row['entity_id'], row['attribute']) == (report['entity_id'], report['attribute'])]
    elif frame.percept.subject and frame.percept.subject != 'user':
        required_current = [(item, row) for item, row in current
                            if frame.percept.subject in {row['entity_id'], row.get('entity', '')}]
    if problem and not required_current and not outcomes:
        if not frame.percept.subject or frame.percept.subject == 'user':
            missing.append('subject')
        missing.append('current_state_or_result')

    grounded_current = any(any(s.kind == 'user_turn' for s in item.sources)
                           for item, _ in required_current)
    learned = any(dict(r.values).get('expected_result') == 'success'
                  and len({s.id for s in r.sources if s.kind == 'outcome'}) >= 2
                  for r in frame.relevant_strategies)
    developed = any(r.kind in {'preference', 'opinion', 'interest'}
                    and len({s.id for s in r.sources}) >= 2 for r in frame.relevant_self)
    surprise = 'surprise' in frame.attention.reason_codes
    if grounded_current:
        reasons.append('grounded_current')
    if corroborated:
        reasons.append('corroborated_observation')
    if learned:
        reasons.append('learned_strategy')
    if developed:
        reasons.append('developed_self')
    if history:
        reasons.append('relevant_history')
        if not current:
            uncertainty.append('historical_only')
    if outcomes:
        reasons.append('linked_outcome')
    if frame.objective:
        reasons.append('active_objective')
    if surprise:
        uncertainty.append('prediction_surprise')
        reasons.append('prediction_surprise')
    if missing:
        uncertainty.append('required_evidence_missing')
        reasons.append('required_evidence_missing')
    if conflicts:
        uncertainty.append('conflicting_evidence')
        reasons.append('structured_conflict')

    confidence = 'medium'
    if (grounded_current and (corroborated or learned)) or (developed and not problem):
        confidence = 'high'
    if missing or conflicts or (report and not corroborated) or (history and not current):
        confidence = 'low'
    elif surprise:
        confidence = 'medium'

    # Count structural dimensions, not domain importance or attention score.
    record_count = sum(len(getattr(frame, field)) for field in (
        'situation', 'relevant_world', 'relevant_self', 'relevant_memory',
        'relevant_experience', 'relevant_strategies', 'relevant_goals', 'linked_evidence',
    ))
    dimensions = sum((bool(history), bool(frame.relevant_strategies),
                      bool(frame.linked_evidence), len(relations) >= 2, record_count > 4))
    complexity = 'high' if conflicts or missing or dimensions >= 3 else 'medium' if dimensions else 'low'
    verification = bool(conflicts or missing or (surprise and frame.objective))
    mode = ('verify' if verification else 'deliberate' if complexity != 'low'
            else 'normal' if current or frame.objective or developed else 'fast')
    if not reasons:
        reasons.append('simple_context')
    result = DeliberationAssessment(
        frame.profile_id, frame.percept.id,
        'frame_' + hashlib.sha256(frame.to_json().encode('utf-8')).hexdigest(),
        mode, complexity, confidence, tuple(uncertainty), tuple(conflicts),
        tuple(missing), verification, tuple(reasons), frame.created_at,
    )
    return result if len(result.to_json().encode('utf-8')) <= ASSESSMENT_MAX_BYTES else None
