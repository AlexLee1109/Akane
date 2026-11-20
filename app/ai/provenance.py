"""Attribution shared by the three layers, without transferring ownership."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SourceAttribution:
    owner: Literal["user", "l1", "l2", "l3", "external"]
    source_id: str
    uri: str | None = None
