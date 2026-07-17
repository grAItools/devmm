"""Public conformance entry points (design §9): `mr_conformance` runs the
memory-resource contract suite against a caller-supplied factory, and
`dlpack_conformance` runs the DLPack producer contract suite for tensors on a
device. Both raise `AssertionError` naming the first violated contract, so
third-party MR/runtime authors get the same guarantees devmm's own suite
enforces.

The individual check functions are stdlib-only (no pytest/hypothesis), so the
entry points work in any environment that can import devmm; the pytest mixin
in `devmm.testing._mr_conformance` reuses them per test method.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import inspect
import weakref
from collections.abc import Callable, Iterator

from devmm._core.device import Device
from devmm._core.layout import Aligned, RowMajor
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.stream import CpuStream, Stream
from devmm._core.tensor import Tensor, empty
from devmm._dlpack._abi import (
    DLPACK_FLAG_BITMASK_READ_ONLY,
    DLPACK_VERSION,
    DLManagedTensor,
    DLManagedTensorVersioned,
    DLTensor,
)

_ALIGNMENTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
_SWEEP_NBYTES = (1, 17, 255, 4096)

WriteFn = Callable[[int, bytes], None]
ReadFn = Callable[[int, int], bytes]


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextlib.contextmanager
def _expect(exc_type: type[BaseException], description: str) -> Iterator[None]:
    """Fail with `description` unless the body raises `exc_type`."""
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"{description}: expected {exc_type.__name__}, but nothing was raised")


def _pattern(nbytes: int) -> bytes:
    # Period 251 (prime) so the pattern can never line up with power-of-two
    # allocation granularities and mask an addressing bug.
    return bytes(i % 251 for i in range(nbytes))


def host_write(ptr: int, data: bytes) -> None:
    """Default `write`: a host `memmove` (valid for host-addressable memory)."""
    ctypes.memmove(ptr, data, len(data))


def host_read(ptr: int, nbytes: int) -> bytes:
    """Default `read`: a host byte copy (valid for host-addressable memory)."""
    return ctypes.string_at(ptr, nbytes)


def live_count(mr: DeviceMemoryResource) -> int:
    """Read the `_debug_live_count()` testing hook devmm MRs expose."""
    hook: Callable[[], object] | None = getattr(mr, "_debug_live_count", None)
    if not callable(hook):
        raise AssertionError(f"{mr!r} does not expose the _debug_live_count() testing hook")
    count = hook()
    if not isinstance(count, int):
        raise AssertionError(f"_debug_live_count() must return an int, got {count!r}")
    return count


def check_write_then_read(
    mr: DeviceMemoryResource, stream: Stream, write: WriteFn, read: ReadFn
) -> None:
    data = _pattern(1024)
    ptr = mr.allocate(len(data), stream)
    write(ptr, data)
    _ensure(read(ptr, len(data)) == data, "a written pattern must read back byte-exact")
    mr.deallocate(ptr, len(data), stream)


def check_no_aliasing(
    mr: DeviceMemoryResource, stream: Stream, write: WriteFn, read: ReadFn
) -> None:
    first = mr.allocate(256, stream)
    write(first, b"\xaa" * 256)
    second = mr.allocate(256, stream)
    write(second, b"\x55" * 256)
    _ensure(
        read(first, 256) == b"\xaa" * 256,
        "live allocations must not alias: writing the second clobbered the first",
    )
    mr.deallocate(first, 256, stream)
    _ensure(
        read(second, 256) == b"\x55" * 256,
        "live allocations must not alias: freeing the first disturbed the second",
    )
    mr.deallocate(second, 256, stream)


def check_guaranteed_alignment(mr: DeviceMemoryResource, stream: Stream) -> None:
    alignment = mr.guaranteed_alignment()
    _ensure(alignment >= 1, f"guaranteed_alignment() must be >= 1, got {alignment}")
    for nbytes in (1, 3, 17, 255, 4096):
        ptr = mr.allocate(nbytes, stream)
        _ensure(
            ptr % alignment == 0,
            f"guaranteed_alignment() is dishonest: pointer {ptr:#x} is not "
            f"{alignment}-byte aligned",
        )
        mr.deallocate(ptr, nbytes, stream)


def check_requested_alignment(
    mr: DeviceMemoryResource,
    stream: Stream,
    write: WriteFn,
    read: ReadFn,
    *,
    nbytes: int,
    alignment: int,
) -> None:
    ptr = mr.allocate(nbytes, stream)
    _ensure(
        ptr % alignment == 0,
        f"requested alignment not delivered: pointer {ptr:#x} is not {alignment}-byte aligned",
    )
    # Writing and reading the whole span catches off-by-one over-allocation
    # at the aligned offset.
    data = _pattern(nbytes)
    write(ptr, data)
    _ensure(read(ptr, nbytes) == data, "an aligned allocation must hold its full span")
    mr.deallocate(ptr, nbytes, stream)


def check_bookkeeping(mr: DeviceMemoryResource, stream: Stream) -> None:
    live = [(mr.allocate(nbytes, stream), nbytes) for nbytes in range(0, 64, 7)]
    _ensure(
        live_count(mr) == len(live),
        "_debug_live_count() must report every outstanding allocation",
    )
    for ptr, nbytes in live:
        mr.deallocate(ptr, nbytes, stream)
    _ensure(live_count(mr) == 0, "bookkeeping must be empty after paired alloc/free")


def check_double_free(mr: DeviceMemoryResource, stream: Stream) -> None:
    ptr = mr.allocate(64, stream)
    mr.deallocate(ptr, 64, stream)
    with _expect(ValueError, "a double-free must raise ValueError"):
        mr.deallocate(ptr, 64, stream)


def check_foreign_free(mr: DeviceMemoryResource, stream: Stream) -> None:
    ptr = mr.allocate(64, stream)
    with _expect(ValueError, "freeing an unknown pointer must raise ValueError"):
        mr.deallocate(ptr + 1, 64, stream)
    # The refused free must not have disturbed the real allocation.
    mr.deallocate(ptr, 64, stream)
    _ensure(live_count(mr) == 0, "a refused foreign free must leave the allocation freeable")


def check_zero_byte(mr: DeviceMemoryResource, stream: Stream) -> None:
    first = mr.allocate(0, stream)
    second = mr.allocate(0, stream)
    _ensure(first != 0 and second != 0, "zero-byte allocations must return non-null pointers")
    # Concurrently live zero-byte allocations must still be distinct
    # pointers, or the caller's bookkeeping (and the MR's own) collides.
    _ensure(first != second, "live zero-byte allocations must have distinct pointers")
    mr.deallocate(first, 0, stream)
    mr.deallocate(second, 0, stream)
    _ensure(live_count(mr) == 0, "zero-byte allocations must free cleanly")


def check_negative_size(mr: DeviceMemoryResource, stream: Stream) -> None:
    with _expect(ValueError, "a negative allocation size must raise ValueError"):
        mr.allocate(-1, stream)


def _accepts_alignment(factory: Callable[..., DeviceMemoryResource]) -> bool:
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "alignment"
            and parameter.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        for parameter in parameters
    )


def mr_conformance(
    mr_factory: Callable[..., DeviceMemoryResource],
    *,
    stream_factory: Callable[[], Stream] | None = None,
    write: Callable[[int, bytes], None] | None = None,
    read: Callable[[int, int], bytes] | None = None,
) -> None:
    """Run the `DeviceMemoryResource` conformance suite (design §9).

    `mr_factory` must return a fresh MR per call; if it accepts an
    `alignment: int` keyword, the requested-alignment sweep runs too. The
    suite pins the allocate/deallocate contract: byte-exact writes/reads
    through returned pointers, no aliasing between live allocations,
    alignment honesty, bookkeeping hygiene via the MR's
    ``_debug_live_count()`` testing hook, misuse detection (double-free,
    foreign-pointer free, negative sizes raise ``ValueError``), and the
    zero-byte contract. The first violated contract raises
    ``AssertionError``.

    The default `write`/`read` dereference pointers on the host, which is
    only valid for host-addressable memory (CPU MRs, pinned host memory).
    For device MRs, pass `write`/`read` backed by the runtime's `memcpy` and
    a `stream_factory` returning streams on the MR's device.
    """
    streams = stream_factory if stream_factory is not None else CpuStream
    write_fn = write if write is not None else host_write
    read_fn = read if read is not None else host_read

    check_write_then_read(mr_factory(), streams(), write_fn, read_fn)
    check_no_aliasing(mr_factory(), streams(), write_fn, read_fn)
    check_guaranteed_alignment(mr_factory(), streams())
    if _accepts_alignment(mr_factory):
        for alignment in _ALIGNMENTS:
            mr = mr_factory(alignment=alignment)
            stream = streams()
            for nbytes in _SWEEP_NBYTES:
                check_requested_alignment(
                    mr, stream, write_fn, read_fn, nbytes=nbytes, alignment=alignment
                )
    check_bookkeeping(mr_factory(), streams())
    check_double_free(mr_factory(), streams())
    check_foreign_free(mr_factory(), streams())
    check_zero_byte(mr_factory(), streams())
    check_negative_size(mr_factory(), streams())


# Independent C-API bindings (prototype-from-symbol) instead of attribute
# access on `ctypes.pythonapi`: the attribute-cached function objects are
# process-global, so setting `argtypes` there would leak into every other
# user of the same symbol.
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)

_VERSIONED_NAME = b"dltensor_versioned"
_LEGACY_NAME = b"dltensor"


def _managed_versioned(capsule: object) -> DLManagedTensorVersioned:
    _ensure(
        bool(_capsule_is_valid(capsule, _VERSIONED_NAME)),
        f"a versioned export must be a capsule named {_VERSIONED_NAME.decode()!r}",
    )
    address = _capsule_get_pointer(capsule, _VERSIONED_NAME)
    return ctypes.cast(address, ctypes.POINTER(DLManagedTensorVersioned)).contents


def _managed_legacy(capsule: object) -> DLManagedTensor:
    _ensure(
        bool(_capsule_is_valid(capsule, _LEGACY_NAME)),
        f"a legacy export must be a capsule named {_LEGACY_NAME.decode()!r}",
    )
    address = _capsule_get_pointer(capsule, _LEGACY_NAME)
    return ctypes.cast(address, ctypes.POINTER(DLManagedTensor)).contents


def _check_dl_tensor(dl: DLTensor, tensor: Tensor) -> None:
    expected_device = tensor.__dlpack_device__()
    _ensure(
        (dl.device.device_type, dl.device.device_id) == expected_device,
        f"exported DLDevice {(dl.device.device_type, dl.device.device_id)} != "
        f"__dlpack_device__() {expected_device}",
    )
    dtype = (dl.dtype.code, dl.dtype.bits, dl.dtype.lanes)
    expected_dtype = (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes)
    _ensure(dtype == expected_dtype, f"exported DLDataType {dtype} != {expected_dtype}")
    ndim = len(tensor.shape)
    _ensure(dl.ndim == ndim, f"exported ndim {dl.ndim} != {ndim}")
    shape = tuple(dl.shape[i] for i in range(ndim))
    _ensure(shape == tensor.shape, f"exported shape {shape} != {tensor.shape}")
    strides = tuple(dl.strides[i] for i in range(ndim))
    _ensure(
        strides == tensor.layout.strides,
        f"exported strides {strides} != layout strides {tensor.layout.strides}",
    )
    _ensure(dl.byte_offset == 0, f"exported byte_offset must be 0, got {dl.byte_offset}")
    if 0 in tensor.shape:
        _ensure(dl.data is None, "a zero-size export must carry data == NULL")
    else:
        expected_data = tensor.buffer.ptr + tensor.offset * tensor.dtype.itemsize
        _ensure(
            dl.data == expected_data,
            f"exported data {dl.data} != buffer address {expected_data}",
        )


def dlpack_conformance(device: Device) -> None:
    """Run the DLPack producer conformance suite for tensors on `device`
    (design §7, §9).

    Tensors are allocated with `devmm.empty` through the registry's current
    MR for `device` (scope one with `using_memory_resource` to target a
    specific MR), so a working runtime/MR pair for the device is required.
    Every check is metadata-level — capsule names, struct fields, refusal
    `BufferError`s, holder lifetime — so the suite runs against any device
    without dereferencing device memory. The first violated contract raises
    ``AssertionError``.
    """
    if not isinstance(device, Device):
        raise TypeError(f"device must be a devmm Device, got {device!r}")
    expected_pair = (int(device.type), device.index)

    tensor = empty((2, 3), "float32", device=device, layout=RowMajor())
    _ensure(
        tensor.__dlpack_device__() == expected_pair,
        f"__dlpack_device__() returned {tensor.__dlpack_device__()}, expected {expected_pair}",
    )

    # Capsules are bound to locals for the duration of every struct read:
    # dropping an unconsumed capsule runs its destructor, which frees the
    # exported block the struct view points into.
    capsule = tensor.__dlpack__(max_version=DLPACK_VERSION)
    versioned = _managed_versioned(capsule)
    _ensure(
        (versioned.version.major, versioned.version.minor) == DLPACK_VERSION,
        "a max_version >= the producer version must stamp the producer version",
    )
    _ensure(versioned.flags == 0, "a writable export must carry no flags")
    _check_dl_tensor(versioned.dl_tensor, tensor)

    capsule = tensor.__dlpack__(max_version=(1, 0))
    clamped = _managed_versioned(capsule)
    _ensure(
        (clamped.version.major, clamped.version.minor) == (1, 0),
        "the stamped version must be min(producer, consumer)",
    )

    capsule = tensor.__dlpack__()
    _check_dl_tensor(_managed_legacy(capsule).dl_tensor, tensor)

    padded = empty((4, 3), "float32", device=device, layout=Aligned(RowMajor()))
    capsule = padded.__dlpack__(max_version=DLPACK_VERSION)
    _check_dl_tensor(_managed_versioned(capsule).dl_tensor, padded)

    read_only = Tensor(tensor.buffer, tensor.dtype, tensor.shape, tensor.layout, read_only=True)
    capsule = read_only.__dlpack__(max_version=DLPACK_VERSION)
    _ensure(
        _managed_versioned(capsule).flags & DLPACK_FLAG_BITMASK_READ_ONLY != 0,
        "read_only=True must set DLPACK_FLAG_BITMASK_READ_ONLY on versioned exports",
    )
    with _expect(BufferError, "a read-only tensor must refuse a legacy-only consumer"):
        read_only.__dlpack__()

    capsule = tensor.__dlpack__(max_version=DLPACK_VERSION, dl_device=expected_pair)
    _managed_versioned(capsule)
    foreign_pair = (2, 0) if device.type != 2 else (1, 0)
    with _expect(BufferError, "a mismatched dl_device must be refused"):
        tensor.__dlpack__(max_version=DLPACK_VERSION, dl_device=foreign_pair)

    capsule = tensor.__dlpack__(max_version=DLPACK_VERSION, copy=False)
    _managed_versioned(capsule)
    with _expect(BufferError, "copy=True must be refused (exports are zero-copy)"):
        tensor.__dlpack__(max_version=DLPACK_VERSION, copy=True)

    with _expect(BufferError, "a malformed max_version must be refused"):
        tensor.__dlpack__(max_version=(1,))  # type: ignore[arg-type]

    # -1 declines synchronization on every platform; the capsule must still
    # be produced.
    capsule = tensor.__dlpack__(max_version=DLPACK_VERSION, stream=-1)
    _managed_versioned(capsule)

    zero = empty((0, 3), "float32", device=device, layout=RowMajor())
    capsule = zero.__dlpack__(max_version=DLPACK_VERSION)
    _check_dl_tensor(_managed_versioned(capsule).dl_tensor, zero)
    del capsule

    # Deleting an unconsumed capsule must release the holder: with the
    # producing tensor dropped too, nothing may keep the tensor alive.
    doomed = empty((2, 2), "float32", device=device, layout=RowMajor())
    doomed_ref = weakref.ref(doomed)
    capsule = doomed.__dlpack__(max_version=DLPACK_VERSION)
    del doomed, capsule
    gc.collect()
    _ensure(
        doomed_ref() is None,
        "deleting an unconsumed capsule must release the exported tensor",
    )

    freed = empty((2, 2), "float32", device=device, layout=RowMajor())
    freed.buffer.free()
    with _expect(BufferError, "exporting a freed buffer must be refused"):
        freed.__dlpack__(max_version=DLPACK_VERSION)

    # The refusal checks above raised through frames that reference the
    # suite's tensors (exception -> traceback -> frame cycles); collect so
    # the suite leaves no allocation pending in the caller's MR.
    del tensor, padded, read_only, zero, freed
    gc.collect()
