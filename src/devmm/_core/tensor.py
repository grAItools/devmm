"""`Tensor`: the minimal DLPack producer (dtype + shape + layout + offset over a `DeviceBuffer`)
plus the `empty()`/`empty_like()` factories (§3.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from devmm._core.buffer import DeviceBuffer
from devmm._core.device import Device, DeviceType
from devmm._core.dtypes import DType
from devmm._core.layout import DeviceOptimal, Layout, LayoutPolicy
from devmm._core.memory_resource import DeviceMemoryResource
from devmm._core.registry import get_current_memory_resource
from devmm._core.stream import CpuStream, Stream
from devmm._dlpack.export import to_capsule
from devmm._runtimes._discovery import runtime_for

if TYPE_CHECKING:
    # Annotation-only: `types.CapsuleType` is 3.13+, and typing_extensions
    # must stay out of the zero-dependency runtime (design §8).
    from typing_extensions import CapsuleType

_CPU_DEVICE = Device(DeviceType.CPU)
_DEFAULT_LAYOUT_POLICY = DeviceOptimal()


class Tensor:
    """A minimal DLPack producer over a `DeviceBuffer` (design §3.8).

    dtype + shape + strides + element offset, exactly the two protocol
    methods, and introspection properties. Deliberately *not* an array
    library: no indexing, no arithmetic, no casting — consumers get a
    zero-copy view via ``xp.from_dlpack(tensor)``.
    """

    buffer: DeviceBuffer
    dtype: DType
    shape: tuple[int, ...]
    layout: Layout
    offset: int
    read_only: bool

    def __init__(
        self,
        buffer: DeviceBuffer,
        dtype: DType,
        shape: tuple[int, ...],
        layout: Layout,
        *,
        offset: int = 0,
        read_only: bool = False,
    ) -> None:
        shape = tuple(shape)
        layout.validate(shape, dtype.itemsize)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset must be a non-negative element count, got {offset!r}")
        needed = offset * dtype.itemsize + layout.required_nbytes
        if needed > buffer.nbytes:
            raise ValueError(
                f"layout needs {needed} bytes (offset {offset} elements + "
                f"{layout.required_nbytes} addressed) but the buffer holds only {buffer.nbytes}"
            )
        self.buffer = buffer
        self.dtype = dtype
        self.shape = shape
        self.layout = layout
        self.offset = offset
        self.read_only = read_only

    @property
    def device(self) -> Device:
        return self.buffer.device

    @property
    def strides(self) -> tuple[int, ...]:
        """Element strides (DLPack convention), straight from the layout."""
        return self.layout.strides

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.buffer.device.__dlpack_device__()

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        return to_capsule(
            self, stream=stream, max_version=max_version, dl_device=dl_device, copy=copy
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, offset={self.offset}, read_only={self.read_only})"
        )


def _default_stream(device: Device) -> Stream:
    """The stream used when a factory caller passes none.

    CPU work is synchronous (a single no-op stream); every other device gets
    its runtime's platform default stream (design §4.1), so a device no
    runtime serves raises `RuntimeUnavailableError`.
    """
    if device.type is DeviceType.CPU:
        return CpuStream(device)
    return runtime_for(device).default_stream(device)


def _aligned_element_offset(ptr: int, itemsize: int, alignment: int) -> int:
    """Best-effort element offset moving `ptr` onto an `alignment` boundary.

    Element offsets advance in `itemsize` steps, so the address residue
    modulo gcd(itemsize, alignment) is fixed: a weakly-aligned MR can return
    a `ptr` that is not even itemsize-aligned, in which case no offset lands
    on the boundary and 0 is returned — the tensor then starts at the buffer
    base, aligned only as well as the MR delivered."""
    misalignment = -ptr % alignment
    offset, remainder = divmod(misalignment, itemsize)
    return 0 if remainder else offset


def empty(
    shape: tuple[int, ...],
    dtype: object,
    *,
    device: Device = _CPU_DEVICE,
    layout: Layout | LayoutPolicy = _DEFAULT_LAYOUT_POLICY,
    mr: DeviceMemoryResource | None = None,
    stream: Stream | None = None,
) -> Tensor:
    """Allocate an uninitialized `Tensor` (design §3.8).

    `dtype` is anything `DType.from_any` accepts. The layout policy (or a
    pre-resolved, shape-validated `Layout`) drives the strides; `mr` defaults
    to the device's current registry resource; the allocation is over-aligned
    only when `mr.guaranteed_alignment()` cannot honor the layout's base
    alignment (design §3.6). Over-alignment is best-effort: landing exactly
    on the base boundary needs an MR pointer that is at least
    itemsize-aligned (see `_aligned_element_offset`).
    """
    resolved_dtype = DType.from_any(dtype)
    shape = tuple(shape)
    if mr is None:
        mr = get_current_memory_resource(device)
    elif mr.device != device:
        raise ValueError(f"mr lives on {mr.device} but device= names {device}; pass matching ones")
    if stream is None:
        stream = _default_stream(device)
    if isinstance(layout, LayoutPolicy):
        resolved = layout(shape, resolved_dtype, device)
    else:
        layout.validate(shape, resolved_dtype.itemsize)
        resolved = layout
    pad = 0
    if resolved.required_nbytes and mr.guaranteed_alignment() < resolved.base_alignment:
        # One extra alignment span leaves room to slide the tensor start onto
        # the next base-alignment boundary.
        pad = resolved.base_alignment
    buffer = DeviceBuffer(resolved.required_nbytes + pad, mr=mr, stream=stream)
    offset = 0
    if pad:
        offset = _aligned_element_offset(
            buffer.ptr, resolved_dtype.itemsize, resolved.base_alignment
        )
    return Tensor(buffer, resolved_dtype, shape, resolved, offset=offset)


def _dtype_like(dtype: object) -> DType:
    """Coerce a duck-typed dtype the way `empty_like` needs (design §3.8).

    Falls back to the trailing identifier of ``str(dtype)`` because the Array
    API constrains dtype objects almost not at all — array-api-strict's, for
    instance, expose neither ``.kind`` nor ``.itemsize`` but print as the
    (namespaced) canonical spelling.
    """
    try:
        return DType.from_any(dtype)
    except (TypeError, ValueError):
        return DType.from_string(str(dtype).rsplit(".", 1)[-1])


def _device_like(obj: Any) -> Device:
    dlpack_device = getattr(obj, "__dlpack_device__", None)
    if dlpack_device is None:
        raise TypeError(
            f"{type(obj).__name__!r} object has no __dlpack_device__; pass device= explicitly"
        )
    device_type, index = dlpack_device()
    return Device(DeviceType(int(device_type)), int(index))


def empty_like(
    obj: Any,
    *,
    dtype: object | None = None,
    device: Device | None = None,
    layout: Layout | LayoutPolicy = _DEFAULT_LAYOUT_POLICY,
    mr: DeviceMemoryResource | None = None,
    stream: Stream | None = None,
) -> Tensor:
    """Allocate a fresh `Tensor` shaped like `obj` (design §3.8).

    `obj` is duck-typed through `__dlpack_device__()` and the Array API
    `.shape`/`.dtype` attributes — the producing library is never imported.
    Keyword arguments override what `obj` reports; the rest match `empty`.
    """
    shape = tuple(obj.shape)
    resolved_dtype = _dtype_like(obj.dtype if dtype is None else dtype)
    if device is None:
        device = _device_like(obj)
    return empty(shape, resolved_dtype, device=device, layout=layout, mr=mr, stream=stream)
