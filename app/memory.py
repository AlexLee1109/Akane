
"""Memory module - now using async MemoryStore with background writes.

All functions are re-exported from memory_store for backward compatibility.
"""
import atexit

# Import everything from memory_store
from app.memory_store import (
    # Data access
    MEMORY_PATH,  # also export path
    get_store,
    reload_from_disk,
    flush_now,
    get_all,
    get_relationship_context,
    get_recent_mood,
    top_observations,
    format_for_prompt,
    analyze_conversation_context,

    # Mutations
    record_interaction,
    add,
    touch,
    forget,
    archive_activity,
    delete,
    search,

    # Constants
    NEUTRAL,
    POSITIVE,
    NEGATIVE,

    # Lifecycle
    shutdown,
)

# Register shutdown hook to flush any remaining writes
atexit.register(shutdown)

# For backward compatibility, also provide these if anything imports them directly
__all__ = [
    "MEMORY_PATH",
    "get_store",
    "reload_from_disk",
    "flush_now",
    "get_all",
    "get_relationship_context",
    "get_recent_mood",
    "top_observations",
    "format_for_prompt",
    "analyze_conversation_context",
    "record_interaction",
    "add",
    "touch",
    "forget",
    "archive_activity",
    "delete",
    "search",
    "NEUTRAL",
    "POSITIVE",
    "NEGATIVE",
    "shutdown",
]
