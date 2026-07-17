"""Testing utilities that run without hardware (design §9).

`RecordingMemoryResource` is the suite's allocator test double: deterministic
fake pointers, a complete call log, and misuse detection (double-free,
foreign-free, size-mismatch) raised as `RecordingMisuseError`.
"""

from devmm.testing._recording import RecordingMemoryResource, RecordingMisuseError

__all__ = [
    "RecordingMemoryResource",
    "RecordingMisuseError",
]
