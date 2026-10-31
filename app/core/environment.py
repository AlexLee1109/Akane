"""Pure sandbox transitions and evidence. Store owns persistence and execution.

Choices are parameters for the existing inspect/continue/verify actions, not a
second action dispatcher. Nothing here executes code, reads arbitrary paths,
contacts a network, generates dialogue, or assigns preferences.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import uuid

from app.core.state import WorldChange, WorldEntity, WorldSource, WorldState
from app.core.perception import percept_id


PRODUCER = 'virtual-environment'
ACTIVITIES = ('puzzle', 'room', 'exploration')
RECEIPT_LIMIT = 32
VIRTUAL_KIND = 'virtual_environment'
# Curated repository asset metadata; reading metadata does not view the media.
CATALOG = (
    ('story', 'Akane story artwork', 'image/jpeg', 'website/public/assets/akane-story.jpg'),
    ('night', 'Akane night thumbnail', 'image/webp', 'website/public/assets/akane-night-thumb.webp'),
    ('logo', 'Akane logo', 'image/png', 'website/public/assets/akane-logo-192.png'),
)


def wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(wire(value).encode('utf-8')).hexdigest()


def action_occurrence(choice):
    return 'virtual_action_' + digest([choice.instance_id, choice.revision])


def learning_context(choice, state):
    """Observable task conditions that can change an operation's outcome.

    This describes environment rules, not tastes. It never runs a transition
    to reveal its answer during deliberation. Unrelated room state is excluded.
    """
    if choice.activity == 'puzzle':
        return (state['round'] % 3,)
    if choice.activity == 'room' and choice.operation == 'open':
        return (state['unlocked'],)
    return ()


def information_gain(observation, choice):
    """One expected knowledge unit for an unread inspection/new location.

    Only legal affordances can promise this; reversible manipulation and
    starting more puzzles are not intrinsically informative rewards.
    """
    if choice.action == 'inspect':
        return 1.0
    facts = dict(observation.facts)
    if choice.operation == 'move' and choice.target not in facts.get('visited', '').split(','):
        return 1.0
    return 0.0


def goal_targets(choice):
    """Named effects/dependencies of legal affordances, without utility weights."""
    if choice.activity == 'room' and choice.operation == 'take' and choice.target == 'key':
        return 'key chest'
    return choice.target + ' ' + choice.value


def task_progress(observation, choice):
    """A bounded prospective step in the activity's declared objective.

    These are causal affordances, not enjoyment or a claimed successful result.
    Reversible lamp cycling and revisiting a known location are not progress.
    """
    if choice.activity == 'puzzle':
        return float(choice.operation == 'submit')
    if choice.activity == 'room':
        return float(choice.operation in {'take', 'unlock', 'open'})
    return information_gain(observation, choice)


@dataclass(frozen=True, slots=True)
class EnvironmentChoice:
    instance_id: str
    activity: str
    revision: int
    action: str
    operation: str
    target: str = ''
    value: str = ''

    @property
    def id(self):
        return ':'.join((self.operation, self.target, self.value))

    @property
    def procedure(self):
        return ' '.join(x for x in (self.action, 'virtual', self.activity,
                                     self.operation, self.target, self.value) if x)


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    activity: str
    instance_id: str
    revision: int
    facts: tuple[tuple[str, str], ...]
    choices: tuple[EnvironmentChoice, ...]

    @property
    def content(self):
        return f'virtual {self.activity} ' + wire(dict(self.facts))

    @property
    def source_id(self):
        return 'virtual_observation_' + digest(asdict(self))


@dataclass(frozen=True, slots=True)
class EnvironmentChange:
    """A command, never caller-supplied state or a caller-supplied result."""
    activity: str
    expected_revision: int = -1
    episode_id: str = ''


def initial_state(activity):
    if activity == 'puzzle':
        return dict(round=0, inspected=False, solved=False, attempts=0)
    if activity == 'room':
        return dict(seen=[], key=False, unlocked=False, opened=False, lamp=False)
    if activity == 'exploration':
        return dict(location='index', visited=['index'], read=[])
    raise ValueError('Unknown virtual activity.')


def initial_environment(profile_id, activity):
    return dict(instance_id='virtual_' + digest(profile_id)[:16] + '_' + uuid.uuid4().hex,
                revision=0, state=initial_state(activity), receipts=[])


def observation(activity, row):
    state, revision, instance = row['state'], row['revision'], row['instance_id']
    choices = []

    def add(action, operation, target='', value=''):
        choices.append(EnvironmentChoice(instance, activity, revision, action, operation, target, value))

    if activity == 'puzzle':
        left, right = ((3, 5), (4, 7), (2, 9))[state['round'] % 3]
        facts = dict(solved=str(state['solved']), attempts=str(state['attempts']))
        if not state['inspected']:
            add('inspect', 'inspect', 'problem')
        elif not state['solved']:
            facts.update(rule='add the two numbers', numbers=f'{left},{right}')
            for answer in sorted({left * right, left + right}):
                add('verify', 'submit', 'answer', str(answer))
        else:
            add('continue', 'next', 'problem')
    elif activity == 'room':
        facts = dict(objects='desk,chest,lamp')
        for target in ('desk', 'chest', 'lamp'):
            if target not in state['seen']:
                add('inspect', 'inspect', target)
        if 'desk' in state['seen']:
            facts['key'] = 'held' if state['key'] else 'on desk'
            if not state['key']:
                add('continue', 'take', 'key')
        if state['key'] and not state['unlocked']:
            add('continue', 'unlock', 'chest')
        if 'chest' in state['seen']:
            facts['chest'] = 'open' if state['opened'] else 'unlocked' if state['unlocked'] else 'locked'
            if not state['opened']:
                add('continue', 'open', 'chest')  # An executable attempt can fail.
        if 'lamp' in state['seen']:
            facts['lamp'] = 'on' if state['lamp'] else 'off'
            add('continue', 'switch', 'lamp', 'off' if state['lamp'] else 'on')
    elif activity == 'exploration':
        location = state['location']
        facts = dict(location=location, visited=','.join(state['visited']), read=','.join(state['read']),
                     scope='metadata only; media not viewed')
        if location == 'gallery':
            for item, title, mime, path in CATALOG:
                if item not in state['read']:
                    add('inspect', 'read', item)
                else:
                    facts[item] = mime  # Full inspected metadata is in the receipt.
        for target in (('gallery',) if location == 'index' else ('index',)):
            add('continue', 'move', target)
    else:
        raise ValueError('Unknown virtual activity.')
    return EnvironmentObservation(activity, instance, revision, tuple(facts.items()), tuple(choices))


def transition(activity, state, choice):
    """Evaluate a legal operation. Invalid preconditions never become Outcomes."""
    row = dict(instance_id=choice.instance_id, revision=choice.revision, state=state)
    if choice not in observation(activity, row).choices:
        raise ValueError('Illegal virtual action or unsatisfied precondition.')
    updated = deepcopy(state)
    result, detail = 'success', ''
    if activity == 'puzzle':
        if choice.operation == 'inspect':
            updated['inspected'] = True
            detail = 'Inspected the addition problem.'
        elif choice.operation == 'next':
            updated = initial_state(activity)
            updated['round'] = state['round'] + 1
            detail = 'Opened the next addition problem.'
        else:
            left, right = ((3, 5), (4, 7), (2, 9))[state['round'] % 3]
            updated['attempts'] += 1
            updated['solved'] = int(choice.value) == left + right
            result = 'success' if updated['solved'] else 'failure'
            detail = f'Submitted {choice.value}; expected {left + right}; {result}.'
    elif activity == 'room':
        if choice.operation == 'inspect':
            updated['seen'].append(choice.target)
            detail = f'Inspected {choice.target}.'
        elif choice.operation == 'take':
            updated['key'] = True
            detail = 'Took the virtual key from the desk.'
        elif choice.operation == 'unlock':
            updated['unlocked'] = True
            detail = 'Unlocked the virtual chest with the held key.'
        elif choice.operation == 'open':
            updated['opened'] = state['unlocked']
            result = 'success' if updated['opened'] else 'failure'
            detail = 'Opened the virtual chest.' if updated['opened'] else 'Chest remained closed: locked.'
        else:
            updated['lamp'] = choice.value == 'on'
            detail = f'Set the virtual lamp {choice.value}.'
    elif choice.operation == 'move':
        updated['location'] = choice.target
        if choice.target not in updated['visited']:
            updated['visited'].append(choice.target)
        detail = f'Moved to the virtual {choice.target}.'
    else:
        updated['read'].append(choice.target)
        item = next(item for item in CATALOG if item[0] == choice.target)
        detail = f'Read metadata: {item[1]}; {item[2]}; {item[3]}. Media not viewed.'
    return updated, result, detail


def advance(profile_id, activity, row, episode, now):
    choice = episode.environment
    if choice is None or (choice.activity, choice.instance_id, choice.revision) != (
            activity, row['instance_id'], row['revision']):
        raise ValueError('Stale or mismatched virtual action.')
    state, result, detail = transition(activity, row['state'], choice)
    revision = row['revision'] + 1
    receipt = dict(id='virtual_result_' + digest([profile_id, row['instance_id'], revision]),
        episode_id=episode.id, source_percept_id=episode.source_percept_id,
        occurrence_id=action_occurrence(choice),
        percept_id=percept_id(profile_id, 'task', 'completion', PRODUCER, action_occurrence(choice)),
        choice=asdict(choice), before=deepcopy(row['state']), after=state,
        before_hash=digest(row['state']), after_hash=digest(state), result=result,
        detail=detail, timestamp=now)
    return dict(instance_id=row['instance_id'], revision=revision, state=state,
                receipts=[*row['receipts'], receipt][-RECEIPT_LIMIT:]), receipt


def validate_environment(environments, profile_id):
    if not isinstance(environments, dict) or set(environments) - set(ACTIVITIES):
        raise ValueError('Invalid virtual environment map.')
    for activity, row in environments.items():
        if not isinstance(row, dict) or set(row) != {'instance_id', 'revision', 'state', 'receipts'}:
            raise ValueError('Invalid virtual environment structure.')
        instance = row['instance_id']
        prefix = 'virtual_' + digest(profile_id)[:16] + '_'
        if (not isinstance(instance, str) or not instance.startswith(prefix)
                or len(instance) != len(prefix) + 32
                or any(c not in '0123456789abcdef' for c in instance[len(prefix):])):
            raise ValueError('Invalid virtual environment owner or instance.')
        revision = row['revision']
        if type(revision) is not int or not 0 <= revision <= 2**53:
            raise ValueError('Invalid virtual environment revision.')
        validate_activity_state(activity, row['state'])
        receipts = row['receipts']
        if not isinstance(receipts, list) or len(receipts) != min(revision, RECEIPT_LIMIT):
            raise ValueError('Invalid bounded virtual receipt history.')
        previous, timestamp = None, 0
        for number, receipt in enumerate(receipts, revision - len(receipts)):
            if not isinstance(receipt, dict) or set(receipt) != {
                'id', 'episode_id', 'source_percept_id', 'occurrence_id', 'percept_id', 'choice', 'before', 'after',
                'before_hash', 'after_hash', 'result', 'detail', 'timestamp',
            }:
                raise ValueError('Invalid virtual result evidence.')
            choice = EnvironmentChoice(**receipt['choice'])
            validate_activity_state(activity, receipt['before'])
            if (choice.instance_id, choice.activity, choice.revision) != (instance, activity, number):
                raise ValueError('Invalid virtual result attribution.')
            after, result, detail = transition(activity, receipt['before'], choice)
            if (receipt['after'] != after or receipt['result'] != result or receipt['detail'] != detail
                    or receipt['before_hash'] != digest(receipt['before']) or receipt['after_hash'] != digest(after)
                    or receipt['id'] != 'virtual_result_' + digest([profile_id, instance, number + 1])
                    or receipt['occurrence_id'] != action_occurrence(choice)
                    or receipt['percept_id'] != percept_id(profile_id, 'task', 'completion', PRODUCER, action_occurrence(choice))
                    or receipt['source_percept_id'] != percept_id(profile_id, 'runtime', 'event', PRODUCER,
                        observation(activity, dict(instance_id=instance, revision=number, state=receipt['before'])).source_id)
                    or previous is not None and previous != receipt['before']):
                raise ValueError('Virtual result does not verify against its transition.')
            if (type(receipt['timestamp']) not in (int, float) or not math.isfinite(receipt['timestamp'])
                    or receipt['timestamp'] < timestamp
                    or not isinstance(receipt['episode_id'], str) or not receipt['episode_id'].startswith('episode_')
                    or not isinstance(receipt['source_percept_id'], str)
                    or not receipt['source_percept_id'].startswith('percept_')):
                raise ValueError('Invalid virtual result provenance or time.')
            previous, timestamp = after, receipt['timestamp']
        if receipts and row['state'] != receipts[-1]['after'] or not receipts and row['state'] != initial_state(activity):
            raise ValueError('Virtual state disagrees with its result history.')


def validate_activity_state(activity, state):
    expected = initial_state(activity)
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Invalid virtual activity state.')
    for key, value in state.items():
        if type(value) is not type(expected[key]):
            raise ValueError('Invalid virtual state type.')
    if activity == 'puzzle':
        if not 0 <= state['round'] <= 2**53 or not 0 <= state['attempts'] <= 2**53:
            raise ValueError('Invalid virtual puzzle counters.')
        if state['solved'] and (not state['inspected'] or not state['attempts']):
            raise ValueError('Unverified puzzle completion.')
    elif activity == 'room':
        if (len(set(state['seen'])) != len(state['seen']) or set(state['seen']) - {'desk', 'chest', 'lamp'}
                or state['key'] and 'desk' not in state['seen'] or state['unlocked'] and not state['key']
                or state['opened'] and not state['unlocked']):
            raise ValueError('Invalid virtual room preconditions.')
    elif (state['location'] not in {'index', 'gallery'} or not state['visited']
          or len(set(state['visited'])) != len(state['visited']) or set(state['visited']) - {'index', 'gallery'}
          or state['visited'][0] != 'index' or state['location'] not in state['visited']
          or len(set(state['read'])) != len(state['read']) or set(state['read']) - {x[0] for x in CATALOG}
          or state['read'] and 'gallery' not in state['visited']):
        raise ValueError('Invalid virtual exploration state.')


def world_changes(profile_id, activity, row, now):
    source = row['receipts'][-1]['id'] if row['receipts'] else row['instance_id']
    sources = (WorldSource('environment', source),)
    entity = WorldEntity(row['instance_id'], profile_id, 'virtual ' + activity, now, now, sources, VIRTUAL_KIND)
    # World is a labelled projection. Canonical simulation truth is isolated in
    # profile.environment and cannot be written through dialogue World evidence.
    values = {'revision': str(row['revision']), 'observations': wire(dict(observation(activity, row).facts))}
    return (WorldChange('upsert', entity), *(WorldChange('upsert', WorldState(
        entity.id + ':' + key, profile_id, entity.id, key, value, 1.0, now, now, sources,
    )) for key, value in values.items()))


def situation_changes(profile_id, activity, world, source, now):
    from app.core.mind import _stable_id
    from app.core.situation import akane_entity
    sources = (WorldSource('environment', source),)
    entity = akane_entity(world, profile_id)
    changes = []
    if entity is None:
        entity = WorldEntity(_stable_id('world_entity', profile_id, 'akane'), profile_id,
                             'Akane', now, now, sources)
        changes.append(WorldChange('upsert', entity))
    for slot in ('focus', 'intention'):
        changes.append(WorldChange('upsert', WorldState(
            _stable_id('situation', profile_id, entity.id, slot), profile_id, entity.id,
            slot, 'virtual ' + activity, 1.0, now, now, sources)))
    return tuple(changes)
