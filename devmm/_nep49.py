"""NEP-49 interop shared by both integration directions (design §5.1, §6):
the ctypes mirror of NumPy's `PyDataMem_Handler`, access to
`PyDataMem_GetHandler`/`PyDataMem_SetHandler` through NumPy's exported
`_ARRAY_API` function table, and the supported-version guard.

The entry points are not dlopen-visible symbols of the multiarray extension
(hidden visibility), so they are reached the way NumPy's own C consumers
reach them: as slots of the `void**` table the `_ARRAY_API` capsule carries.
"""

from __future__ import annotations

import ctypes
import importlib
from types import ModuleType
from typing import Any

from devmm._runtimes.base import RuntimeUnavailableError

# The NEP-49 callback prototypes, verbatim from `numpy/ndarraytypes.h`:
#   void *malloc(void *ctx, size_t size)
#   void *calloc(void *ctx, size_t nelem, size_t elsize)
#   void *realloc(void *ctx, void *ptr, size_t new_size)
#   void  free(void *ctx, void *ptr, size_t size)
MallocFunc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
CallocFunc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)
ReallocFunc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
FreeFunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)


class PyDataMemAllocator(ctypes.Structure):
    """Mirror of NumPy's `PyDataMemAllocator`: a context pointer plus the
    four allocation entry points."""

    ctx: int | None

    _fields_ = [
        ("ctx", ctypes.c_void_p),
        ("malloc", MallocFunc),
        ("calloc", CallocFunc),
        ("realloc", ReallocFunc),
        ("free", FreeFunc),
    ]


class PyDataMemHandler(ctypes.Structure):
    """Mirror of NumPy's `PyDataMem_Handler`: a 127-byte name (sized so the
    struct stays pointer-aligned), a one-byte ABI version (currently 1) and
    the allocator vtable."""

    name: bytes
    version: int
    allocator: PyDataMemAllocator

    _fields_ = [
        ("name", ctypes.c_char * 127),
        ("version", ctypes.c_uint8),
        ("allocator", PyDataMemAllocator),
    ]


#: The capsule name NumPy requires of every handler capsule.
HANDLER_CAPSULE_NAME = b"mem_handler"

#: The `PyDataMem_Handler.version` the mirror above understands.
HANDLER_ABI_VERSION = 1

#: Supported NumPy range `[low, high)`: NEP-49 landed in 1.22 and the
#: handler ABI (and the API-table indices below) have been stable through
#: 2.x. A NumPy outside the range fails `require_supported_numpy` until the
#: mirror is re-verified against it (design §5.1).
SUPPORTED_NUMPY_RANGE: tuple[tuple[int, int], tuple[int, int]] = ((1, 22), (3, 0))

# Slots of the NEP-49 entry points in NumPy's append-only C-API table
# (`numpy/core/code_generators/numpy_api.py`), fixed since 1.22.
_SET_HANDLER_INDEX = 304
_GET_HANDLER_INDEX = 305

# Independent C-API bindings (prototype-from-symbol) instead of attribute
# access on `ctypes.pythonapi`: the attribute-cached function objects are
# process-global, so setting `argtypes` there would leak into every other
# user of the same symbol (the `devmm._dlpack.export` convention).
py_inc_ref = ctypes.PYFUNCTYPE(None, ctypes.py_object)(("Py_IncRef", ctypes.pythonapi))
py_is_initialized = ctypes.PYFUNCTYPE(ctypes.c_int)(("Py_IsInitialized", ctypes.pythonapi))

PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
capsule_new = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, PyCapsuleDestructor
)(("PyCapsule_New", ctypes.pythonapi))
# Raw-pointer variants for capsule destructors: wrapping a dying capsule in
# `py_object` would resurrect it (the `devmm._dlpack.export` convention).
capsule_is_valid_raw = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
capsule_pointer_raw = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)
_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)

_GetHandlerFunc = ctypes.PYFUNCTYPE(ctypes.py_object)
_SetHandlerFunc = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.py_object)


def parsed_version(version: str) -> tuple[int, int]:
    """`(major, minor)` of a NumPy version string."""
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        raise RuntimeError(f"cannot parse the NumPy version string {version!r}") from None


def require_supported_numpy(module: ModuleType) -> None:
    """Range guard (design §5.1): refuse a NumPy whose NEP-49 ABI this
    module has not been verified against."""
    version = parsed_version(module.__version__)
    low, high = SUPPORTED_NUMPY_RANGE
    if not (low <= version < high):
        raise RuntimeError(
            f"NumPy {module.__version__} is outside the range devmm's NEP-49 "
            f"interop is pinned to ({low[0]}.{low[1]} <= version < "
            f"{high[0]}.{high[1]}); the PyDataMem_Handler mirror is verified "
            "only inside it (design §5.1)"
        )


def _numpy_module() -> ModuleType:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "numpy is not importable; install numpy to use devmm's NEP-49 interop"
        ) from exc


def _multiarray_umath(module: ModuleType) -> ModuleType:
    # NumPy 2.x moved the extension to `numpy._core`; 1.x keeps `numpy.core`.
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError(
        f"NumPy {module.__version__} exposes no _multiarray_umath extension "
        "under numpy._core or numpy.core; cannot reach the NEP-49 entry points"
    )


class Nep49Api:
    """The NEP-49 entry points of the running NumPy.

    Both entry points return a *new* reference, which ctypes' `py_object`
    restype conversion takes ownership of (verified empirically by the
    refcount-drift test in `tests/test_integrations_numpy.py`) — so the
    wrappers must not adjust refcounts themselves.
    """

    def __init__(self, module: ModuleType, get_handler: Any, set_handler: Any) -> None:
        self.module = module
        self._get_handler = get_handler
        self._set_handler = set_handler

    def get_handler(self) -> Any:
        """The currently installed handler capsule."""
        return self._get_handler()

    def set_handler(self, capsule: Any) -> Any:
        """Install `capsule` (name ``mem_handler``); returns the previous
        handler capsule."""
        return self._set_handler(capsule)


def handler_pointer(capsule: Any) -> Any:
    """The `PyDataMem_Handler` a handler capsule carries, as a ctypes
    pointer. The capsule must stay alive while the view is used."""
    address = _capsule_pointer(capsule, HANDLER_CAPSULE_NAME)
    return ctypes.cast(address, ctypes.POINTER(PyDataMemHandler))


def load_api() -> Nep49Api:
    """Bind the NEP-49 entry points of the running NumPy (range-guarded)."""
    module = _numpy_module()
    require_supported_numpy(module)
    umath = _multiarray_umath(module)
    table_address = _capsule_pointer(umath._ARRAY_API, None)
    table = ctypes.cast(table_address, ctypes.POINTER(ctypes.c_void_p))
    return Nep49Api(
        module,
        _GetHandlerFunc(table[_GET_HANDLER_INDEX]),
        _SetHandlerFunc(table[_SET_HANDLER_INDEX]),
    )
