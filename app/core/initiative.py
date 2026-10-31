"""Bounded initiative policy over an existing Phase 4 frame; no model access.

The journal is operational evidence (observed triggers and attempted actions),
never learned Self, a claimed experience, or proof that an action succeeded.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.attention import _relevant, _terms
from app.core.perception import _text, PERCEPT_CONTENT_BYTES


JOURNAL_LIMIT = 64
SCHEDULE_LIMIT = 64
RETENTION_SECONDS = 7 * 86400
FOLLOW_UP_TTL = 3600
COOLDOWN_SECONDS = 900
WINDOW_SECONDS = 3600
WINDOW_LIMIT = 3
RECENT_INTERACTION_SECONDS = 60
GENERATION_TOKENS = 128
GENERATION_SECONDS = 8
GENERATION_WINDOW_LIMIT = 2


def empty_initiative():
    return dict(preferences=dict(enabled=True, availability='unknown', quiet_start=22, quiet_end=8),
                journal=[], schedules=[])


def validate_preferences(row):
    if (not isinstance(row, dict) or set(row) != {'enabled', 'availability', 'quiet_start', 'quiet_end'}
            or type(row['enabled']) is not bool
            or row['availability'] not in ('unknown', 'available', 'busy', 'away')
            or any(type(row[k]) is not int or not 0 <= row[k] <= 23 for k in ('quiet_start', 'quiet_end'))):
        raise ValueError('Invalid initiative preferences.')


def validate_initiative(row):
    """Strict persistence validation; corrupt safety state never resets silently."""
    if not isinstance(row, dict) or set(row) != {'preferences', 'journal', 'schedules'}:
        raise ValueError('Invalid initiative state.')
    validate_preferences(row['preferences'])
    if isinstance(row['schedules'], list) and sum(r.get('status') == 'pending' for r in row['schedules']
                                                if isinstance(r, dict)) > 16:
        raise ValueError('Initiative schedule capacity exceeded.')
    for group, limit in (('journal', JOURNAL_LIMIT), ('schedules', SCHEDULE_LIMIT)):
        rows = row[group]
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError('Initiative capacity exceeded.')
        ids = set()
        for item in rows:
            keys = ({'id', 'fingerprint', 'subject', 'content', 'created_at', 'expires_at', 'action',
                     'status', 'episode_id', 'outcome_id', 'generation_at', 'offered_turn_id'}
                    if group == 'journal' else {'id', 'at', 'subject', 'goal_id', 'status'})
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError('Invalid initiative record.')
            if not _text(item['id']) or item['id'] in ids or not _text(item['subject'], optional=True):
                raise ValueError('Invalid initiative identity.')
            ids.add(item['id'])
            times = ('created_at', 'expires_at', 'generation_at') if group == 'journal' else ('at',)
            if any(type(item[k]) not in (int, float) or not math.isfinite(item[k]) or item[k] < 0 for k in times):
                raise ValueError('Invalid initiative time.')
            if group == 'journal':
                if (item['action'] not in ('act', 'follow_up')
                        or item['status'] not in ('reserved', 'pending', 'offered', 'completed',
                                                  'awaiting_result', 'cancelled', 'expired')
                        or item['expires_at'] < item['created_at']
                        or not _text(item['fingerprint'])
                        or not _text(item['content'], PERCEPT_CONTENT_BYTES, optional=True)
                        or any(not _text(item[k], optional=True) for k in
                               ('episode_id', 'outcome_id', 'offered_turn_id'))):
                    raise ValueError('Invalid initiative outcome.')
            elif item['status'] not in ('pending', 'fired', 'cancelled') or not _text(item['goal_id'], optional=True):
                raise ValueError('Invalid initiative schedule.')


def quiet(preferences, now, timezone):
    hour = datetime.fromtimestamp(now, ZoneInfo(timezone)).hour
    start, end = preferences['quiet_start'], preferences['quiet_end']
    return (start <= hour < end if start < end else hour >= start or hour < end) if start != end else False


def communication_block(preferences, journal, now, timezone):
    if not preferences['enabled']:
        return 'disabled'
    if preferences['availability'] in ('busy', 'away'):
        return 'user_unavailable'
    if quiet(preferences, now, timezone):
        return 'quiet_hours'
    recent = [r for r in journal if r['action'] == 'follow_up' and now - r['created_at'] < WINDOW_SECONDS]
    if any(now - r['created_at'] < COOLDOWN_SECONDS for r in recent):
        return 'cooldown'
    if len(recent) >= WINDOW_LIMIT:
        return 'frequency_budget'
    return ''


def fingerprint(percept):
    # Distinct IDs with identical observations cannot manufacture new interest.
    data = (percept.source, percept.producer, percept.subject, ' '.join(percept.content.casefold().split()))
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    action: str = 'wait'
    reason_codes: tuple[str, ...] = ('no_meaningful_event',)
    subject: str = ''


def evaluate(frame, assessment, decision, *, state, now, timezone, last_interaction=None,
             recent_turns=(), deadline=False):
    """One frame, fixed bounded records, deterministic WAIT-first admission.

    Selection does not upgrade attention, create goals, or infer an event from
    text. Existing legal virtual actions/handoffs remain owned by ActionSelection.
    Only a significant, relevant observation may also warrant a conversation.
    """
    p = frame.percept
    journal, preferences = state['journal'], state['preferences']
    if p.source == 'user':
        return InitiativeDecision(reason_codes=('foreground_owned',))
    if not preferences['enabled']:
        return InitiativeDecision(reason_codes=('disabled',))
    if not 0 <= now - p.timestamp < 300:
        return InitiativeDecision(reason_codes=('stale_event',))
    if any(r['id'] == p.id for r in journal):
        return InitiativeDecision(reason_codes=('already_considered',))
    subject = next((dict(r.values).get('topic', '') for r in frame.relevant_goals), '')
    subject = subject or next((dict(r.values).get('topic', '') for r in frame.relevant_self
                              if r.kind in {'curiosity', 'interest'}), '') or p.subject
    conversation = follow_up(frame, assessment, state=state, now=now, timezone=timezone,
        last_interaction=last_interaction, recent_turns=recent_turns, deadline=deadline)
    if conversation.action == 'follow_up':
        return conversation
    # Current phase action selection remains the authority for useful legal work.
    actionable = decision and decision.selected_action in {'inspect', 'verify', 'continue'}
    if actionable:
        return InitiativeDecision('act', ('existing_action',), subject)
    return conversation


def follow_up(frame, assessment, *, state, now, timezone, last_interaction=None,
              recent_turns=(), deadline=False):
    p = frame.percept
    journal = state['journal']
    subject = next((dict(r.values).get('topic', '') for r in frame.relevant_goals), '')
    anchors = [*frame.relevant_goals, *(r for r in frame.relevant_self
               if r.kind in {'curiosity', 'interest'} and len({s.id for s in r.sources}) >= 2
               and dict(r.values).get('status', 'active') == 'active'
               and (r.kind != 'interest' or dict(r.values).get('stance') == 'positive')),
               *(r for r in frame.relevant_experience if r.sources)]
    subject = subject or next((dict(r.values).get('topic', '') for r in anchors), '') or p.subject
    meaningful = (deadline or p.source in {'tool', 'task'}
                  or p.source == 'world' and bool(p.event_id) and 'novelty' in frame.attention.reason_codes
                  or 'corroborated_observation' in assessment.reason_codes)
    if not meaningful or not (anchors or deadline):
        return InitiativeDecision()
    if assessment.confidence == 'low':
        return InitiativeDecision(reason_codes=('low_confidence',))
    if any(r['fingerprint'] == fingerprint(p) for r in journal):
        return InitiativeDecision(reason_codes=('repeated_observation',))
    event_terms = _terms(p.content)
    if any(t.role == 'assistant' and t.created_at >= p.timestamp
           and event_terms and event_terms <= _terms(t.content) for t in recent_turns):
        return InitiativeDecision(reason_codes=('already_discussed',))
    block = communication_block(state['preferences'], journal, now, timezone)
    if block:
        return InitiativeDecision(reason_codes=(block,))
    # A deadline is explicitly authorized. Other events should not cut across a
    # recent exchange; they can be noticed again if genuinely new evidence arrives.
    if not deadline and last_interaction is not None and now - last_interaction < RECENT_INTERACTION_SECONDS:
        return InitiativeDecision(reason_codes=('recent_interaction',))
    return InitiativeDecision('follow_up', ('relevant_trusted_event',), subject)


def relevant_offer(row, query):
    return _relevant(_terms(row['subject']), _terms(query), ())
