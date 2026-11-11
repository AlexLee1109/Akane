"""Compatibility import for the retired V3 module name; V2 owns all turns."""

import sys
from app.core import session

sys.modules[__name__] = session
