"""JSONSummaryStore's segment codec/capture/commit helpers; no separate backend."""
from dataclasses import asdict, fields
import hashlib
import json

from ..contracts import GenerationInfo
from ..provenance import SourceAttribution
from ..conversation.cancellation import GenerationCancelled
from .compaction import _summary_dir, _encode as encode_legacy, _compression
from .compaction_contracts import CompactionLimits, SummarySource
from .json_store import _date, _identifier, _keys, _publish, _read, _stored_uuid, _timestamp, _unique_object
from .segment_contracts import SEGMENT_LABEL, SegmentCapture, SegmentCandidate, SummarySegment, SegmentManifest
from .store import ContinuityConflict
from .summary_data import data_json, sources_for


def parse_segment(raw, maximum):
    if not isinstance(raw, str) or len(raw) > maximum:
        raise ValueError('Segment JSON exceeds its character allowance.')
    data = json.loads(raw, object_pairs_hook=_unique_object)
    _keys(data, {'text'})
    if not isinstance(data['text'], str) or not data['text'].strip():
        raise ValueError('Segment summary text must be nonempty; no claim/source fields are accepted.')
    return data['text']


def segment_content(text):
    return data_json({'label': SEGMENT_LABEL, 'text': text})


def range_digest(data, start, end):
    return hashlib.sha256(json.dumps(data['turns'][2*start:2*end], ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def base_identity(legacy):
    return (legacy.checkpoint_id, hashlib.sha256(data_json(encode_legacy(legacy)).encode()).hexdigest(),
            len(legacy.covered_turn_ids)) if legacy else (None, None, 0)


def encode(manifest):
    def source(s):
        return dict(alias=s.alias, turn_id=str(s.turn_id), role=s.role, created_at=_timestamp(s.created_at),
                    attribution=asdict(s.attribution), outcome=s.outcome)
    return dict(schema_version=1, kind='original-source-segments', profile_id=str(manifest.profile_id),
                session_id=str(manifest.session_id), session_created_at=_timestamp(manifest.session_created_at),
                base_checkpoint_id=str(manifest.base_checkpoint_id) if manifest.base_checkpoint_id else None,
                base_checkpoint_digest=manifest.base_checkpoint_digest, base_exchanges=manifest.base_exchanges,
                segments=[dict(segment_id=str(s.segment_id), start=s.start, end=s.end, source_revision=s.source_revision,
                               covered_turn_ids=[str(t) for t in s.covered_turn_ids], source_digest=s.source_digest,
                               sources=[source(v) for v in s.sources], limits=asdict(s.limits),
                               content={'label': SEGMENT_LABEL, 'text': s.candidate.text},
                               provenance={'owner':'l3', 'origin':'model-generated', 'scope':'aggregate-segment',
                                           'model':s.candidate.model, 'template_sha256':s.candidate.template_sha256,
                                           'generated_at':_timestamp(s.candidate.generated_at),
                                           'generation':asdict(s.candidate.generation)},
                               tokens={'source':s.candidate.source_tokens, 'projection':s.candidate.summary_tokens})
                          for s in manifest.segments])


def decode(data):
    _keys(data, {'schema_version','kind','profile_id','session_id','session_created_at','base_checkpoint_id',
                 'base_checkpoint_digest','base_exchanges','segments'})
    if type(data['schema_version']) is not int or data['schema_version'] != 1 or data['kind'] != 'original-source-segments':
        raise ValueError('Invalid segment manifest schema.')
    if type(data['base_exchanges']) is not int or data['base_exchanges'] < 0 or not isinstance(data['segments'], list):
        raise ValueError('Invalid segment base/collection.')
    segments=[]
    for item in data['segments']:
        _keys(item, {'segment_id','start','end','source_revision','covered_turn_ids','source_digest','sources','limits',
                     'content','provenance','tokens'})
        if any(type(item[key]) is not int or item[key] < 0 for key in ('start','end','source_revision')):
            raise ValueError('Segment source ordinals/revision must be nonnegative integers.')
        if not isinstance(item['covered_turn_ids'], list) or not isinstance(item['sources'], list):
            raise ValueError('Invalid segment coverage/source array.')
        _keys(item['limits'], {f.name for f in fields(CompactionLimits)})
        limits=CompactionLimits(**item['limits'])
        sources=[]
        for source in item['sources']:
            _keys(source, {'alias','turn_id','role','created_at','attribution','outcome'})
            _keys(source['attribution'], {'owner','source_id','uri'})
            sources.append(SummarySource(source['alias'],_stored_uuid(source['turn_id']),source['role'],
                                         _date(source['created_at']),SourceAttribution(**source['attribution']),source['outcome']))
        _keys(item['content'], {'label','text'})
        if item['content']['label'] != SEGMENT_LABEL:
            raise ValueError('Aggregate segment trust label changed.')
        text=parse_segment(data_json({'text':item['content']['text']}),limits.max_characters)
        meta=item['provenance']
        _keys(meta, {'owner','origin','scope','model','template_sha256','generated_at','generation'})
        _keys(meta['generation'], {f.name for f in fields(GenerationInfo)})
        if (meta['owner'],meta['origin'],meta['scope']) != ('l3','model-generated','aggregate-segment'):
            raise ValueError('Segment provenance must remain aggregate model data.')
        _keys(item['tokens'], {'source','projection'})
        candidate=SegmentCandidate(text,item['end']-item['start'],_date(meta['generated_at']),meta['model'],
                                   meta['template_sha256'],item['tokens']['source'],item['tokens']['projection'],
                                   GenerationInfo(**meta['generation']))
        segments.append(SummarySegment(_stored_uuid(item['segment_id']),item['start'],item['end'],item['source_revision'],
                                       tuple(_stored_uuid(t) for t in item['covered_turn_ids']),item['source_digest'],
                                       tuple(sources),limits,candidate))
    return SegmentManifest(_stored_uuid(data['profile_id']),_stored_uuid(data['session_id']),_date(data['session_created_at']),
                           _stored_uuid(data['base_checkpoint_id']) if data['base_checkpoint_id'] is not None else None,
                           data['base_checkpoint_digest'],data['base_exchanges'],tuple(segments))


def read_validated(summaries, data, entries, legacy):
    session=summaries._store._session(data)
    identifier, digest, count=base_identity(legacy)
    try:
        with _summary_dir(summaries._store) as directory:
            raw=_read(directory,str(session.conversation_id)+'.segments.json')
    except FileNotFoundError:
        return SegmentManifest(session.profile_id,session.conversation_id,session.created_at,identifier,digest,count)
    manifest=decode(raw)
    if (manifest.profile_id,manifest.session_id,manifest.session_created_at) != (session.profile_id,session.conversation_id,session.created_at):
        raise ContinuityConflict('Segment manifest belongs to another profile/session incarnation.')
    if (manifest.base_checkpoint_id,manifest.base_checkpoint_digest,manifest.base_exchanges) != (identifier,digest,count):
        raise ContinuityConflict('Legacy checkpoint anchor changed beneath committed segments.')
    frontier=count
    seen=set()
    for segment in manifest.segments:
        selected=entries[2*segment.start:2*segment.end]
        if (segment.start != frontier or segment.end <= segment.start or segment.end-segment.start > 8 or segment.end > segment.source_revision
                or segment.source_revision > session.revision or segment.segment_id in seen
                or segment.covered_turn_ids != tuple(e.turn_id for e in selected[::2])
                or len(selected) != 2*(segment.end-segment.start)
                or segment.sources != sources_for(selected,include_diagnostics=True)
                or segment.source_digest != range_digest(data,segment.start,segment.end)):
            raise ContinuityConflict('Segment coverage, outcome, attribution, order or source digest changed.')
        validate_candidate(segment.candidate,segment.limits)
        frontier=segment.end;seen.add(segment.segment_id)
    return manifest


def validate_candidate(candidate, limits):
    if type(candidate.exchanges) is not int or candidate.exchanges < 1:
        raise ValueError('Invalid segment exchange count.')
    parse_segment(data_json({'text':candidate.text}),limits.max_characters)
    if len(segment_content(candidate.text)) > limits.max_characters or not isinstance(candidate.model,str) or not candidate.model:
        raise ValueError('Invalid segment projection or model provenance.')
    _compression(candidate,limits)


def capture(summaries, session_id, *, max_exchanges, limits, cancellation):
    _identifier(session_id)
    if type(max_exchanges) is not int or not 1 <= max_exchanges <= 8:
        raise ValueError('Segment capture bound must be between one and eight exchanges.')
    with summaries._store._locked(cancellation):
        data,entries=summaries._store._load(session_id)
        legacy=summaries._read(session_id)
        if legacy: summaries._validate_source(legacy,data,entries)
        manifest=read_validated(summaries,data,entries,legacy)
        remaining=limits.keep_recent_exchanges; eligible=0
        for index in range(len(entries)-2,-1,-2):
            if entries[index+1].outcome == 'completed': remaining-=1
            if remaining == 0:
                eligible=index//2;break
        start=manifest.frontier;end=min(start+max_exchanges,eligible)
        if end <= start:
            raise ValueError('No uncovered eligible original segment is available.')
        selected=entries[2*start:2*end]
        cancellation.raise_if_cancelled()
        return SegmentCapture(summaries._store._session(data),start,end,selected,range_digest(data,start,end),
                              manifest.frontier_id,manifest.base_checkpoint_id,manifest.base_checkpoint_digest)


def commit(summaries, captured, candidate, *, checkpoint_id, limits, cancellation):
    _identifier(checkpoint_id);validate_candidate(candidate,limits)
    if candidate.exchanges > captured.end-captured.start:
        raise ValueError('Candidate cannot broaden captured segment coverage.')
    with summaries._store._locked(cancellation):
        data,entries=summaries._store._load(captured.session.conversation_id)
        session=summaries._store._session(data)
        if session.profile_id != captured.session.profile_id or session.created_at != captured.session.created_at or session.revision < captured.session.revision:
            raise ContinuityConflict('Segment source namespace/incarnation changed.')
        if captured.entries != entries[2*captured.start:2*captured.end] or captured.digest != range_digest(data,captured.start,captured.end):
            raise ContinuityConflict('Captured original segment changed during generation.')
        legacy=summaries._read(session.conversation_id)
        if legacy: summaries._validate_source(legacy,data,entries)
        manifest=read_validated(summaries,data,entries,legacy)
        if (manifest.base_checkpoint_id,manifest.base_checkpoint_digest) != (captured.base_checkpoint_id,captured.base_checkpoint_digest):
            raise ContinuityConflict('Segment base checkpoint changed.')
        end=captured.start+candidate.exchanges
        selected=entries[2*captured.start:2*end]
        segment=SummarySegment(checkpoint_id,captured.start,end,captured.session.revision,
                               tuple(e.turn_id for e in selected[::2]),range_digest(data,captured.start,end),
                               sources_for(selected,include_diagnostics=True),limits,candidate)
        existing=next((s for s in manifest.segments if s.segment_id==checkpoint_id),None)
        if existing is not None:
            if existing == segment: return existing
            raise ContinuityConflict('Segment identity was reused with different content.')
        if manifest.frontier != captured.start or manifest.frontier_id != captured.frontier_id:
            raise ContinuityConflict('A competing segment advanced the coverage frontier.')
        updated=SegmentManifest(manifest.profile_id,manifest.session_id,manifest.session_created_at,
                                manifest.base_checkpoint_id,manifest.base_checkpoint_digest,manifest.base_exchanges,
                                (*manifest.segments,segment))
        encoded=encode(updated)
        cancellation.raise_if_cancelled()
        with _summary_dir(summaries._store) as directory:
            name=str(session.conversation_id)+'.segments.json'
            try:
                _publish(directory,name,encoded,cancellation)
            except GenerationCancelled as error:
                try: published=_read(directory,name)
                except FileNotFoundError: raise error
                if published != json.loads(json.dumps(encoded)): raise error
        return segment
