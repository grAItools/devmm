"""Probe registry and entry-point loading: cheap platform probes + on-demand loaders, with the
rmm/hipMM name-collision handling and `DEVMM_RUNTIME` override (§4.2).

Discovery holds `(name, probe, loader)` triples. A probe answers "could this
runtime work here?" without importing anything heavyweight; the loader pays
the import/construction cost only when a runtime is actually requested, and
the result is cached process-wide. Third-party runtimes register through the
`devmm.runtimes` entry-point group without touching core (design §4.1).
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata

from devmm._core.device import Device, DeviceType
from devmm._runtimes.base import DeviceRuntime, RuntimeUnavailableError

_ENTRY_POINT_GROUP = "devmm.runtimes"
_ENV_OVERRIDE = "DEVMM_RUNTIME"


@dataclass(frozen=True, slots=True)
class _RuntimeSpec:
    """One discoverable runtime (design §4.1)."""

    name: str
    probe: Callable[[], bool]
    loader: Callable[[], DeviceRuntime]


def _cpu_probe() -> bool:
    # Host memory is always present.
    return True


def _load_cpu() -> DeviceRuntime:
    from devmm._runtimes.cpu import CpuRuntime

    return CpuRuntime()


def _dlopen(name: str) -> object | None:
    """Seam for tests: dlopen `name`, or None when it is not loadable."""
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


_CUDA_DRIVER_LIBRARIES: tuple[str, ...] = (
    ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
)


def _cuda_probe() -> bool:
    # Platform-keyed probe (design §4.2): is an NVIDIA *driver* loadable?
    # Cheap by design — the heavyweight libcudart load happens in the loader.
    return any(_dlopen(name) is not None for name in _CUDA_DRIVER_LIBRARIES)


def _load_cuda() -> DeviceRuntime:
    from devmm._runtimes.cuda import CudaRuntime

    return CudaRuntime()


# libamdhip64 is both the driver-facing and the runtime library on ROCm, so
# the probe and the loader (`devmm._runtimes.rocm`) try the same spellings —
# duplicated here because probes must not import the runtime module (§4.1).
_ROCM_DRIVER_LIBRARIES: tuple[str, ...] = (
    ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll")
    if sys.platform == "win32"
    else ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so")
)


def _rocm_probe() -> bool:
    # Platform-keyed probe (design §4.2): is the AMD HIP runtime loadable?
    # Keying off the platform library — never the ambiguous `rmm` module
    # name hipMM also installs under — is what disambiguates the two GPU
    # stacks.
    return any(_dlopen(name) is not None for name in _ROCM_DRIVER_LIBRARIES)


def _load_rocm() -> DeviceRuntime:
    from devmm._runtimes.rocm import HipRuntime

    return HipRuntime()


# Built-in runtimes, in preference order; entry points are appended after.
_BUILTIN_SPECS: tuple[_RuntimeSpec, ...] = (
    _RuntimeSpec("cpu", _cpu_probe, _load_cpu),
    _RuntimeSpec("cuda", _cuda_probe, _load_cuda),
    _RuntimeSpec("rocm", _rocm_probe, _load_rocm),
)


def _entry_points() -> Iterable[metadata.EntryPoint]:
    """Seam for tests: the installed `devmm.runtimes` entry points."""
    return metadata.entry_points(group=_ENTRY_POINT_GROUP)


def _entry_point_probe() -> bool:
    # An installed entry point is the cheap availability signal; a runtime
    # whose environment check needs real work performs it in its loader and
    # raises RuntimeUnavailableError there.
    return True


def _load_entry_point(entry_point: metadata.EntryPoint) -> DeviceRuntime:
    """Resolve an entry point to a runtime: it must name a zero-argument
    callable (class or factory) returning a `DeviceRuntime`."""
    factory = entry_point.load()
    runtime: DeviceRuntime = factory()
    return runtime


_specs_lock = threading.Lock()
# The installed entry points cannot change for the life of a process, so the
# spec table is scanned once and cached: discovery sits on hot paths (every
# registry default, host copy and DLPack handoff resolves a runtime) and a
# per-call `importlib.metadata` scan of site-packages is ~1000x slower than
# a dict hit.
_specs_cache: tuple[_RuntimeSpec, ...] | None = None


def _clear_spec_cache() -> None:
    """Test seam: force the next discovery call to re-scan entry points."""
    global _specs_cache
    with _specs_lock:
        _specs_cache = None


def _discovered_specs() -> tuple[_RuntimeSpec, ...]:
    """Every registered runtime, built-ins first; the first registration of
    a name wins, so built-ins shadow same-named entry points. Cached
    process-wide (see `_specs_cache`)."""
    global _specs_cache
    with _specs_lock:
        if _specs_cache is None:
            specs = list(_BUILTIN_SPECS)
            seen = {spec.name for spec in specs}
            for entry_point in _entry_points():
                if entry_point.name in seen:
                    continue
                seen.add(entry_point.name)
                specs.append(
                    _RuntimeSpec(
                        entry_point.name,
                        _entry_point_probe,
                        functools.partial(_load_entry_point, entry_point),
                    )
                )
            _specs_cache = tuple(specs)
        return _specs_cache


def _selected_specs() -> tuple[_RuntimeSpec, ...]:
    """The runtimes eligible right now: the probe-passing subset, or exactly
    the `DEVMM_RUNTIME` override — forced, so its probe is skipped (§4.2)."""
    override = os.environ.get(_ENV_OVERRIDE)
    specs = _discovered_specs()
    if override is not None:
        for spec in specs:
            if spec.name == override:
                return (spec,)
        registered = ", ".join(spec.name for spec in specs)
        raise RuntimeUnavailableError(
            f"DEVMM_RUNTIME={override!r} names no registered device runtime "
            f"(registered: {registered}); unset DEVMM_RUNTIME or set it to one "
            "of the registered names"
        )
    return tuple(spec for spec in specs if spec.probe())


_load_lock = threading.Lock()
# name -> constructed runtime: loaders run (and heavyweight imports happen)
# at most once per process.
_loaded: dict[str, DeviceRuntime] = {}


def _load(spec: _RuntimeSpec) -> DeviceRuntime:
    with _load_lock:
        runtime = _loaded.get(spec.name)
        if runtime is None:
            runtime = spec.loader()
            _loaded[spec.name] = runtime
        return runtime


def _normalized_device_type(device: Device | DeviceType | str) -> DeviceType:
    if isinstance(device, str):
        return Device.from_string(device).type
    if isinstance(device, Device):
        return device.type
    return device


def runtime_names() -> tuple[str, ...]:
    """Names of the runtimes available in this environment.

    Probes only — no runtime construction, no heavyweight imports — so
    "what's available?" never pays import costs (design §4.1).
    """
    return tuple(spec.name for spec in _selected_specs())


def available_runtimes() -> tuple[DeviceRuntime, ...]:
    """Probe every registered runtime and load the passing ones (§4.1).

    A probe is only a cheap prediction: a runtime whose loader then finds
    the environment lacking raises `RuntimeUnavailableError` (the loader
    convention, see `_entry_point_probe`) and is skipped, never failing the
    others.
    """
    runtimes = []
    for spec in _selected_specs():
        try:
            runtimes.append(_load(spec))
        except RuntimeUnavailableError:
            continue
    return tuple(runtimes)


def runtime_for(device: Device | DeviceType | str) -> DeviceRuntime:
    """The runtime serving `device`, loaded on first use (design §4.1).

    Accepts a `Device`, a bare `DeviceType`, or a device string ("cpu",
    "cuda:1"). Raises `RuntimeUnavailableError` when no available runtime
    supports the device's type.
    """
    device_type = _normalized_device_type(device)
    specs = _selected_specs()
    for spec in specs:
        try:
            runtime = _load(spec)
        except RuntimeUnavailableError:
            # One unloadable runtime must not poison the search for the
            # others (its device types are unknown until it loads, so it
            # cannot be excluded any earlier).
            continue
        if device_type in runtime.device_types:
            return runtime
    # "registered", not "available": a listed runtime passed its probe (or
    # was forced) but may still have failed to load.
    registered = ", ".join(spec.name for spec in specs) or "none"
    raise RuntimeUnavailableError(
        f"no available device runtime supports {device_type.name.lower()} "
        f"devices (registered: {registered}); install the matching platform "
        f"stack, register a runtime in the {_ENTRY_POINT_GROUP!r} entry-point "
        "group, or set DEVMM_RUNTIME to force a specific runtime"
    )
