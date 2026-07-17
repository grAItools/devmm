"""Capsule construction, the module-global deleter thunks, version negotiation
and the `stream=` consumer handoff (design §7.2/§7.3).

Per capsule there is **one** ctypes allocation laid out as
``[managed struct | shape int64[ndim] | strides int64[ndim]]``. Ownership
chain: a `_Holder` strongly references the producing `Tensor` (hence buffer,
hence MR chain) and that block; ``manager_ctx`` carries a `Py_IncRef`'d
pointer to the holder; the managed deleter decrefs it, and CPython's own
refcounting then frees the block with everything it kept alive.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

from devmm._core.device import DeviceType
from devmm._dlpack._abi import (
    DLPACK_FLAG_BITMASK_READ_ONLY,
    DLPACK_VERSION,
    DLManagedTensor,
    DLManagedTensorDeleter,
    DLManagedTensorVersioned,
    DLManagedTensorVersionedDeleter,
    DLTensor,
)

if TYPE_CHECKING:
    # Annotation-only: `types.CapsuleType` is 3.13+, and typing_extensions
    # must stay out of the zero-dependency runtime (design §8).
    from typing_extensions import CapsuleType

    from devmm._core.tensor import Tensor

_CAPSULE_NAME_VERSIONED = b"dltensor_versioned"
_CAPSULE_NAME_LEGACY = b"dltensor"

# Independent C-API bindings (prototype-from-symbol) instead of attribute
# access on `ctypes.pythonapi`: the attribute-cached function objects are
# process-global, so setting `argtypes` there would leak into every other
# user of the same symbol.
_py_inc_ref = ctypes.PYFUNCTYPE(None, ctypes.py_object)(("Py_IncRef", ctypes.pythonapi))
_py_dec_ref = ctypes.PYFUNCTYPE(None, ctypes.py_object)(("Py_DecRef", ctypes.pythonapi))
_py_is_initialized = ctypes.PYFUNCTYPE(ctypes.c_int)(("Py_IsInitialized", ctypes.pythonapi))

_PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_capsule_new = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, _PyCapsuleDestructor
)(("PyCapsule_New", ctypes.pythonapi))
# The destructor sees the dying capsule as a raw pointer: wrapping an object
# whose deallocation is in progress in `py_object` would resurrect it.
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


class _Holder:
    """The ``manager_ctx`` target: pins the producing tensor and the exported
    block for exactly as long as any consumer can still reach them
    (design §7.2)."""

    __slots__ = ("block", "tensor")

    def __init__(self, tensor: Tensor, block: ctypes.Array[ctypes.c_char]) -> None:
        self.tensor = tensor
        self.block = block


def _release_managed(
    managed_ptr: Any,
    # Everything the body needs is bound as a default so the deleter stays
    # callable even after this module's globals are torn down at interpreter
    # shutdown.
    _is_initialized: Any = _py_is_initialized,
    _dec_ref: Any = _py_dec_ref,
    _cast: Any = ctypes.cast,
    _py_object: Any = ctypes.py_object,
) -> None:
    """Deleter body shared by both struct flavours: decref the `_Holder`.

    ctypes callbacks acquire the GIL on entry, so consumers may run this from
    any thread (the DLPack deleter requirement); the `Py_IsInitialized` guard
    turns a post-finalization invocation into a no-op instead of a crash.
    """
    if not _is_initialized():
        return
    ctx = managed_ptr.contents.manager_ctx
    if ctx:
        # Null before the decref: a protocol-violating second deleter call
        # then lands on this guard instead of double-decrefing the holder.
        managed_ptr.contents.manager_ctx = None
        _dec_ref(_cast(ctx, _py_object))


# Module-scope references are load-bearing: a garbage-collected CFUNCTYPE
# thunk is a segfault the moment a consumer runs the deleter (design §7.2).
_VERSIONED_DELETER = DLManagedTensorVersionedDeleter(_release_managed)
_LEGACY_DELETER = DLManagedTensorDeleter(_release_managed)

_VERSIONED_POINTER = ctypes.POINTER(DLManagedTensorVersioned)
_LEGACY_POINTER = ctypes.POINTER(DLManagedTensor)


def _destroy_versioned_capsule(
    capsule_ptr: int | None,
    _is_initialized: Any = _py_is_initialized,
    _is_valid: Any = _capsule_is_valid,
    _get_pointer: Any = _capsule_get_pointer,
    _cast: Any = ctypes.cast,
    _pointer_type: Any = _VERSIONED_POINTER,
    _name: bytes = _CAPSULE_NAME_VERSIONED,
) -> None:
    """Capsule destructor: run the managed deleter iff the capsule was never
    consumed (a consumer renames it ``used_dltensor_versioned`` and takes
    ownership of the struct, design §7.2)."""
    if not capsule_ptr or not _is_initialized() or not _is_valid(capsule_ptr, _name):
        return
    managed = _cast(_get_pointer(capsule_ptr, _name), _pointer_type)
    managed.contents.deleter(managed)


def _destroy_legacy_capsule(
    capsule_ptr: int | None,
    _is_initialized: Any = _py_is_initialized,
    _is_valid: Any = _capsule_is_valid,
    _get_pointer: Any = _capsule_get_pointer,
    _cast: Any = ctypes.cast,
    _pointer_type: Any = _LEGACY_POINTER,
    _name: bytes = _CAPSULE_NAME_LEGACY,
) -> None:
    """Legacy twin of `_destroy_versioned_capsule` (name ``used_dltensor``)."""
    if not capsule_ptr or not _is_initialized() or not _is_valid(capsule_ptr, _name):
        return
    managed = _cast(_get_pointer(capsule_ptr, _name), _pointer_type)
    managed.contents.deleter(managed)


_VERSIONED_CAPSULE_DESTRUCTOR = _PyCapsuleDestructor(_destroy_versioned_capsule)
_LEGACY_CAPSULE_DESTRUCTOR = _PyCapsuleDestructor(_destroy_legacy_capsule)

# Immortalize the thunks: a consumer (or an unconsumed capsule) may outlive
# this module's teardown at interpreter shutdown, and the deleter function
# pointers baked into exported structs must stay valid until the very end.
_py_inc_ref(_VERSIONED_DELETER)
_py_inc_ref(_LEGACY_DELETER)
_py_inc_ref(_VERSIONED_CAPSULE_DESTRUCTOR)
_py_inc_ref(_LEGACY_CAPSULE_DESTRUCTOR)


def _negotiate_version(
    max_version: tuple[int, int] | None, read_only: bool
) -> tuple[int, int] | None:
    """Pick the exported flavour: the version to stamp into a
    `DLManagedTensorVersioned`, or None for a legacy `DLManagedTensor`
    (design §7.3)."""
    if max_version is not None:
        message = (
            f"max_version must be None or a (major, minor) pair of "
            f"non-negative ints, got {max_version!r}"
        )
        try:
            major, minor = max_version
        except (TypeError, ValueError):
            raise BufferError(message) from None
        if any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0
            for part in (major, minor)
        ):
            raise BufferError(message)
        if major >= 1:
            return DLPACK_VERSION if (major, minor) >= DLPACK_VERSION else (1, minor)
    if read_only:
        raise BufferError(
            "the legacy DLManagedTensor cannot express the read-only flag; "
            "request max_version >= (1, 0) or export a writable tensor"
        )
    return None


_CUDA_LIKE = frozenset({DeviceType.CUDA, DeviceType.CUDA_HOST, DeviceType.CUDA_MANAGED})
_ROCM_LIKE = frozenset({DeviceType.ROCM, DeviceType.ROCM_HOST})


def _validated_consumer_stream(device_type: DeviceType, stream: int | None) -> int | None:
    """Apply the per-platform stream validation table (design §7.3); return
    the consumer handle the producer must order against, or None when no
    synchronization was requested.

    Per the Array API, `stream=None` means the consumer works on the
    platform's legacy default stream (CUDA: 1, ROCm: 0) and still requires
    ordering; only -1 declines synchronization. CPU consumers have no
    streams, so None/-1 are their whole table.
    """
    if stream is not None:
        if isinstance(stream, bool) or not isinstance(stream, int):
            raise BufferError(
                f"stream must be None or an int per the DLPack protocol, got {stream!r}"
            )
        if stream == -1:  # "do not synchronize", every platform
            return None
    if device_type is DeviceType.CPU:
        if stream is None:
            return None
        raise BufferError(f"CPU consumers must pass stream=None or -1, got {stream}")
    if device_type in _CUDA_LIKE:
        if stream is None:
            return 1  # the legacy default stream
        # 0 is ambiguous between the legacy (1) and per-thread (2) defaults.
        if stream < 1:
            raise BufferError(
                f"CUDA consumer streams are 1 (legacy default), 2 (per-thread "
                f"default) or a raw handle > 2, got {stream}"
            )
        return stream
    if device_type in _ROCM_LIKE:
        if stream is None:
            return 0  # the default stream
        # 1 and 2 are CUDA magic values with no HIP meaning; 0 is the default.
        if stream < 0 or stream in (1, 2):
            raise BufferError(
                f"ROCm consumer streams are 0 (default) or a raw handle > 2, got {stream}"
            )
        return stream
    raise BufferError(f"no stream validation table for device type {device_type!r}")


def _fill_dl_tensor(dl: DLTensor, tensor: Tensor, shape_address: int, ndim: int) -> None:
    if 0 in tensor.shape:
        # Zero-size tensors hold no elements; the protocol exports NULL
        # (design §7.3).
        dl.data = None
    else:
        dl.data = tensor.buffer.ptr + tensor.offset * tensor.dtype.itemsize
    dl.device.device_type = int(tensor.buffer.device.type)
    dl.device.device_id = tensor.buffer.device.index
    dl.ndim = ndim
    dl.dtype.code = tensor.dtype.code
    dl.dtype.bits = tensor.dtype.bits
    dl.dtype.lanes = tensor.dtype.lanes
    # Strides are always emitted explicitly — legal for contiguous tensors
    # too, and it sidesteps shape-relative contiguity questions entirely
    # (design §7.3 skips the strides=NULL shortcut).
    dl.shape = ctypes.cast(shape_address, ctypes.POINTER(ctypes.c_int64))
    dl.strides = ctypes.cast(
        shape_address + ndim * ctypes.sizeof(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    )
    dl.byte_offset = 0


def to_capsule(
    tensor: Tensor,
    *,
    stream: int | None = None,
    max_version: tuple[int, int] | None = None,
    dl_device: tuple[int, int] | None = None,
    copy: bool | None = None,
) -> CapsuleType:
    """Build the DLPack capsule behind `Tensor.__dlpack__` (design §7.2/§7.3).

    Every refusal raises `BufferError` per the protocol: freed buffers,
    cross-device `dl_device`, `copy=True`, malformed versions, and stream
    ints outside the platform validation table.
    """
    buffer = tensor.buffer
    if buffer.closed:
        raise BufferError("cannot export a freed DeviceBuffer (use-after-free)")
    if dl_device is not None and tuple(dl_device) != tensor.__dlpack_device__():
        raise BufferError(
            f"cross-device export is not supported: this tensor lives on "
            f"{buffer.device} (dl_device {tensor.__dlpack_device__()}), the consumer "
            f"asked for {tuple(dl_device)}; copy through your array library instead"
        )
    if copy:
        raise BufferError("copy=True is not supported; devmm exports are always zero-copy")
    version = _negotiate_version(max_version, tensor.read_only)
    consumer_handle = _validated_consumer_stream(buffer.device.type, stream)
    if consumer_handle is not None:
        # Event-based ordering (runtime.make_stream_wait, design §4.1) needs a
        # device runtime; a full producer-stream synchronize is the
        # conservative, always-correct fallback. CPU never reaches this: its
        # table admits no consumer handle.
        buffer.stream.synchronize()

    ndim = len(tensor.shape)
    struct_type: type[DLManagedTensorVersioned] | type[DLManagedTensor]
    struct_type = DLManagedTensorVersioned if version is not None else DLManagedTensor
    struct_size = ctypes.sizeof(struct_type)
    int64_size = ctypes.sizeof(ctypes.c_int64)
    block = ctypes.create_string_buffer(struct_size + 2 * ndim * int64_size)
    block_address = ctypes.addressof(block)
    shape_array = (ctypes.c_int64 * ndim).from_buffer(block, struct_size)
    strides_array = (ctypes.c_int64 * ndim).from_buffer(block, struct_size + ndim * int64_size)
    shape_array[:] = list(tensor.shape)
    strides_array[:] = list(tensor.layout.strides)

    holder = _Holder(tensor, block)
    if version is not None:
        versioned = DLManagedTensorVersioned.from_buffer(block)
        versioned.version.major, versioned.version.minor = version
        versioned.manager_ctx = id(holder)
        versioned.deleter = _VERSIONED_DELETER
        versioned.flags = DLPACK_FLAG_BITMASK_READ_ONLY if tensor.read_only else 0
        _fill_dl_tensor(versioned.dl_tensor, tensor, block_address + struct_size, ndim)
        name, destructor = _CAPSULE_NAME_VERSIONED, _VERSIONED_CAPSULE_DESTRUCTOR
    else:
        legacy = DLManagedTensor.from_buffer(block)
        legacy.manager_ctx = id(holder)
        legacy.deleter = _LEGACY_DELETER
        _fill_dl_tensor(legacy.dl_tensor, tensor, block_address + struct_size, ndim)
        name, destructor = _CAPSULE_NAME_LEGACY, _LEGACY_CAPSULE_DESTRUCTOR

    _py_inc_ref(holder)
    try:
        capsule: CapsuleType = _capsule_new(block_address, name, destructor)
    except BaseException:
        _py_dec_ref(holder)
        raise
    return capsule
