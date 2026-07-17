"""Per-device current-memory-resource registry with strong refs and a contextvars-based scoped
override (§3.4).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType

from devmm._core.device import Device
from devmm._core.memory_resource import DeviceMemoryResource

_lock = threading.Lock()

# Strong references by design: rmm's C++ current-resource map stores raw
# pointers with no lifetime provenance, and its Python layer keeps its own
# dict precisely to fix that; we do the same at this layer (design §3.4).
_registry: dict[Device, DeviceMemoryResource] = {}

# Scoped overrides live in a ContextVar so they compose with threads and
# asyncio tasks; the mapping is immutable so contexts never share mutable
# state.
_NO_OVERRIDES: Mapping[Device, DeviceMemoryResource] = MappingProxyType({})
_overrides: ContextVar[Mapping[Device, DeviceMemoryResource]] = ContextVar(
    "devmm_mr_overrides", default=_NO_OVERRIDES
)


def _unwired_default(device: Device) -> DeviceMemoryResource:
    """Lazy-default seam: device runtimes replace this with their
    `default_memory_resource` lookup (design §4.1). Until one is wired,
    resolution for an unset device fails cleanly.

    The factory is invoked while `_lock` (non-reentrant) is held, so a
    replacement must never call back into `get_current_memory_resource` or
    `set_current_memory_resource` — that would deadlock."""
    raise LookupError(
        f"no current memory resource is set for {device} and no device runtime "
        "default is wired; call set_current_memory_resource() first"
    )


_default_factory: Callable[[Device], DeviceMemoryResource] = _unwired_default


def get_current_memory_resource(device: Device) -> DeviceMemoryResource:
    """Resolve `device`'s current MR: scoped override first, then the
    process-wide registry, then the runtime's lazy default (design §3.4)."""
    override = _overrides.get().get(device)
    if override is not None:
        return override
    with _lock:
        mr = _registry.get(device)
        if mr is None:
            mr = _default_factory(device)
            _registry[device] = mr
        return mr


def set_current_memory_resource(mr: DeviceMemoryResource) -> None:
    """Make `mr` the process-wide current resource for `mr.device`."""
    with _lock:
        _registry[mr.device] = mr


@contextmanager
def using_memory_resource(mr: DeviceMemoryResource) -> Iterator[DeviceMemoryResource]:
    """Scoped override: `mr` is `mr.device`'s current resource inside the
    `with` block (in this thread/task only) and the previous resolution is
    restored on exit, normal or exceptional."""
    token = _overrides.set(MappingProxyType({**_overrides.get(), mr.device: mr}))
    try:
        yield mr
    finally:
        _overrides.reset(token)
