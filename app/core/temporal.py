"""Trusted, queryable clock context. Reading time never emits a percept."""

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import SETTINGS


@dataclass(frozen=True, slots=True)
class TemporalContext:
    now: float
    timezone: str
    local_date: str
    local_time: str
    weekday: str
    day_period: str

    def age(self, timestamp):
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or timestamp < 0 or timestamp > self.now):
            return None
        return self.now - timestamp

    def values(self):
        return tuple((key, str(getattr(self, key))) for key in (
            'now', 'timezone', 'local_date', 'local_time', 'weekday', 'day_period'))


def temporal_context(*, now=None, timezone=SETTINGS.timezone):
    current = time.time() if now is None else now
    if type(current) not in (int, float) or not math.isfinite(current) or current < 0:
        raise ValueError('Clock time must be finite and nonnegative.')
    utc = datetime.fromtimestamp(current, ZoneInfo('UTC'))
    local = utc.astimezone(ZoneInfo(timezone)) if timezone else utc.astimezone()
    hour = local.hour
    period = ('morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17
              else 'evening' if 17 <= hour < 22 else 'night')
    return TemporalContext(float(current), timezone or local.tzname(),
                           local.date().isoformat(), local.isoformat().split('T', 1)[1],
                           local.strftime('%A'), period)


def time_relevant(text):
    return bool(re.search(
        r'\b(what time|what day|what date|timezone|time zone|how long|elapsed|deadline)\b',
        text.casefold(),
    ))
