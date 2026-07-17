"""Testing utilities that run without hardware (design §9).

`RecordingMemoryResource` is the suite's allocator test double: deterministic
fake pointers, a complete call log, and misuse detection (double-free,
foreign-free, size-mismatch) raised as `RecordingMisuseError`.

`mr_conformance` and `dlpack_conformance` are the public conformance entry
points: they run devmm's own contract suites against a third-party memory
resource or a device's DLPack producer path, raising `AssertionError` on the
first violated contract.
"""

from devmm.testing._conformance import dlpack_conformance, mr_conformance
from devmm.testing._recording import RecordingMemoryResource, RecordingMisuseError

__all__ = [
    "RecordingMemoryResource",
    "RecordingMisuseError",
    "dlpack_conformance",
    "mr_conformance",
]
