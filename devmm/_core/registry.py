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
from devmm._runtimes._discovery import runtime_for

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


def _runtime_default(device: Device) -> DeviceMemoryResource:
    """Lazy default: first access for an unset device asks its runtime for
    the default MR (design §3.4, §4.1). Raises `RuntimeUnavailableError`
    when no runtime serves the device.

    The factory is invoked while `_lock` (non-reentrant) is held, so it must
    never call back into `get_current_memory_resource` or
    `set_current_memory_resource` — that would deadlock. Discovery and the
    runtimes' `default_memory_resource` construct MRs without consulting the
    registry."""
    return runtime_for(device).default_memory_resource(device)


_default_factory: Callable[[Device], DeviceMemoryResource] = _runtime_default


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
