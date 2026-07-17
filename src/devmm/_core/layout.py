"""`LayoutPolicy` ABC, the frozen resolved `Layout`, and the shipped policies (RowMajor, ColMajor,
Permuted, Aligned, DeviceOptimal) (§3.6).

Strides are in **elements** (DLPack convention, not bytes). Policies only ever
produce strides that are positive, non-overlapping, and derivable from a
dimension permutation plus innermost-extent padding; `Layout.validate()`
enforces the same deliberate limits on hand-built layouts so that
`required_nbytes` stays well-defined for the exporter.
"""

from __future__ import annotations

import abc
import dataclasses
import itertools
import math

from devmm._core.device import Device, DeviceType
from devmm._core.dtypes import DType


class LayoutPolicy(abc.ABC):
    """An immutable, hashable recipe turning `(shape, dtype, device)` into a
    resolved `Layout` (design §3.6).

    The alignment properties are declared **upper bounds**, not per-call
    values: a dispatching policy (`DeviceOptimal`) may request less for a
    particular device. The exact values live on the produced `Layout`;
    `layout.base_alignment <= policy.base_alignment` always holds.
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def base_alignment(self) -> int:
        """Most bytes of base-pointer alignment the policy may ever request."""

    @property
    @abc.abstractmethod
    def unit_stride_alignment(self) -> int:
        """Most bytes of line-pitch alignment the policy may ever request;
        1 means no padding."""

    @abc.abstractmethod
    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout: ...


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """A concrete, fully resolved memory layout for one `(shape, dtype)`.

    `permutation` orders dimensions outermost -> innermost (C order for
    ndim=3 is `(0, 1, 2)`, F order `(2, 1, 0)`); `strides` are in elements.
    `policy` is best-effort provenance: the policy that produced this layout,
    or `None` for hand-built layouts (e.g. explicit strides from an import
    path).
    """

    permutation: tuple[int, ...]
    strides: tuple[int, ...]
    required_nbytes: int
    base_alignment: int
    policy: LayoutPolicy | None = None

    @property
    def is_contiguous(self) -> bool:
        """True when the strides enumerate offsets densely over the extents
        they imply (unit innermost stride, each outer stride an exact multiple
        of its inner neighbour).

        A `Layout` stores no shape, so contiguity is relative to the extents
        implied by adjacent stride ratios: a layout padded by `Aligned` is
        indistinguishable from a dense layout over the padded extents.
        Shape-relative contiguity ("was padding applied?") needs the shape and
        is answered where a shape is available.
        """
        if not self.permutation:
            return True
        if self.strides[self.permutation[-1]] != 1:
            return False
        return all(
            self.strides[outer] >= self.strides[inner]
            and self.strides[outer] % self.strides[inner] == 0
            for outer, inner in itertools.pairwise(self.permutation)
        )

    def validate(self, shape: tuple[int, ...], itemsize: int) -> None:
        """Raise `ValueError` unless this layout safely addresses `shape`.

        Enforces the deliberate limits of design §3.6 on hand-built layouts:
        strides positive and derivable from a permutation plus padding (unit
        innermost stride, exact multiples outward, each pitch at least the
        inner dimension's extent — hence non-overlapping), and every element
        offset landing inside `required_nbytes`.
        """
        _check_shape(shape)
        if not isinstance(itemsize, int) or itemsize < 1:
            raise ValueError(f"itemsize must be a positive int, got {itemsize!r}")
        ndim = len(shape)
        if len(self.strides) != ndim or len(self.permutation) != ndim:
            raise ValueError(
                f"rank mismatch: shape {shape!r} vs strides {self.strides!r} "
                f"and permutation {self.permutation!r}"
            )
        _check_permutation(self.permutation, ndim)
        for stride in self.strides:
            if not isinstance(stride, int) or stride < 1:
                raise ValueError(f"strides must be positive ints, got {self.strides!r}")
        if not isinstance(self.required_nbytes, int) or self.required_nbytes < 0:
            raise ValueError(
                f"required_nbytes must be a non-negative int, got {self.required_nbytes!r}"
            )
        if not isinstance(self.base_alignment, int) or self.base_alignment < 1:
            raise ValueError(f"base_alignment must be a positive int, got {self.base_alignment!r}")
        if ndim and self.strides[self.permutation[-1]] != 1:
            raise ValueError(
                f"innermost stride must be 1 (element strides derive from a "
                f"permutation plus padding), got {self.strides!r}"
            )
        for outer, inner in itertools.pairwise(self.permutation):
            pitch, remainder = divmod(self.strides[outer], self.strides[inner])
            if remainder:
                raise ValueError(
                    f"stride of dim {outer} is not a multiple of dim {inner}'s: {self.strides!r}"
                )
            if pitch < max(shape[inner], 1):
                raise ValueError(
                    f"strides {self.strides!r} overlap for shape {shape!r}: "
                    f"dim {outer}'s pitch {pitch} is smaller than dim {inner}'s extent"
                )
        if 0 not in shape:
            span = 1 + sum(s * (e - 1) for s, e in zip(self.strides, shape, strict=True))
            if span * itemsize > self.required_nbytes:
                raise ValueError(
                    f"layout addresses {span * itemsize} bytes for shape {shape!r} "
                    f"but required_nbytes is only {self.required_nbytes}"
                )


def _check_shape(shape: tuple[int, ...]) -> None:
    for extent in shape:
        if not isinstance(extent, int) or extent < 0:
            raise ValueError(f"shape extents must be non-negative ints, got {shape!r}")


def _check_permutation(permutation: tuple[int, ...], ndim: int) -> None:
    if sorted(permutation) != list(range(ndim)):
        raise ValueError(f"{permutation!r} is not a permutation of {ndim} dimensions")


def _resolve(
    permutation: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: DType,
    *,
    unit_stride_alignment: int,
    base_alignment: int,
    policy: LayoutPolicy,
) -> Layout:
    """Turn a dimension permutation into concrete element strides and a byte
    size (the shared resolution algorithm of design §3.6).

    The innermost extent is padded up to the smallest count whose byte length
    is a multiple of `unit_stride_alignment`; every outer stride is the exact
    cumulative product of the (padded) inner extents, with zero extents
    treated as 1 so strides stay positive. `required_nbytes` covers the last
    addressable element and is rounded up to `base_alignment` (0 for shapes
    with a zero extent — they hold no elements).
    """
    _check_shape(shape)
    ndim = len(shape)
    _check_permutation(permutation, ndim)
    itemsize = dtype.itemsize
    strides = [0] * ndim
    if ndim:
        innermost = permutation[-1]
        # Smallest element count whose byte length is unit-aligned: a multiple
        # of unit/gcd(itemsize, unit), which also covers itemsizes that do not
        # divide the alignment.
        step = unit_stride_alignment // math.gcd(itemsize, unit_stride_alignment)
        strides[innermost] = 1
        pitch = -(-max(shape[innermost], 1) // step) * step
        for dim in reversed(permutation[:-1]):
            strides[dim] = pitch
            pitch *= max(shape[dim], 1)
    if 0 in shape:
        required_nbytes = 0
    else:
        span = 1 + sum(s * (e - 1) for s, e in zip(strides, shape, strict=True))
        required_nbytes = -(-span * itemsize // base_alignment) * base_alignment
    return Layout(permutation, tuple(strides), required_nbytes, base_alignment, policy)


@dataclasses.dataclass(frozen=True, slots=True)
class RowMajor(LayoutPolicy):
    """C order: identity permutation, no padding."""

    @property
    def base_alignment(self) -> int:
        return 1

    @property
    def unit_stride_alignment(self) -> int:
        return 1

    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout:
        return _resolve(
            tuple(range(len(shape))),
            shape,
            dtype,
            unit_stride_alignment=1,
            base_alignment=1,
            policy=self,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ColMajor(LayoutPolicy):
    """Fortran order: reversed permutation, no padding."""

    @property
    def base_alignment(self) -> int:
        return 1

    @property
    def unit_stride_alignment(self) -> int:
        return 1

    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout:
        return _resolve(
            tuple(reversed(range(len(shape)))),
            shape,
            dtype,
            unit_stride_alignment=1,
            base_alignment=1,
            policy=self,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Permuted(LayoutPolicy):
    """Dimensions laid out in the given outermost -> innermost order, no padding."""

    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_permutation(self.permutation, len(self.permutation))

    @property
    def base_alignment(self) -> int:
        return 1

    @property
    def unit_stride_alignment(self) -> int:
        return 1

    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout:
        if len(self.permutation) != len(shape):
            raise ValueError(
                f"permutation {self.permutation!r} does not match {len(shape)}-d shape {shape!r}"
            )
        return _resolve(
            self.permutation,
            shape,
            dtype,
            unit_stride_alignment=1,
            base_alignment=1,
            policy=self,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Aligned(LayoutPolicy):
    """Pad the innermost extent so every contiguous line starts on a
    `unit_stride_alignment`-byte boundary, and round the allocation size up to
    `base_alignment`.

    The dimension order comes from `inner`; the alignments applied are always
    this wrapper's own, so with nested `Aligned` policies the outermost wins.
    """

    inner: LayoutPolicy
    unit_stride_alignment: int = 128
    base_alignment: int = 256

    def __post_init__(self) -> None:
        if self.unit_stride_alignment < 1 or self.base_alignment < 1:
            raise ValueError(
                f"alignments must be positive, got unit_stride_alignment="
                f"{self.unit_stride_alignment}, base_alignment={self.base_alignment}"
            )

    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout:
        return _resolve(
            self.inner(shape, dtype, device).permutation,
            shape,
            dtype,
            unit_stride_alignment=self.unit_stride_alignment,
            base_alignment=self.base_alignment,
            policy=self,
        )


# Host memory (plain CPU and the pinned-host mirrors) is tuned for CPU access,
# so it gets cache-line alignment; GPU-resident memory (including managed,
# whose performance-critical accessor is the GPU) gets the coalescing line
# pitch and the 256-byte base that CUDA/HIP allocators guarantee anyway.
_HOST_DEVICE_TYPES = frozenset({DeviceType.CPU, DeviceType.CUDA_HOST, DeviceType.ROCM_HOST})
_CPU_CACHELINE = 64
_GPU_UNIT_STRIDE_ALIGNMENT = 128
_GPU_BASE_ALIGNMENT = 256


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceOptimal(LayoutPolicy):
    """Row-major with per-device alignment: GPU-resident memory as
    `Aligned(RowMajor(), 128, 256)`, host-resident memory cache-line aligned.

    The alignment properties report the maxima across devices (upper-bound
    semantics, design §3.6).
    """

    @property
    def base_alignment(self) -> int:
        return _GPU_BASE_ALIGNMENT

    @property
    def unit_stride_alignment(self) -> int:
        return _GPU_UNIT_STRIDE_ALIGNMENT

    def __call__(self, shape: tuple[int, ...], dtype: DType, device: Device) -> Layout:
        if device.type in _HOST_DEVICE_TYPES:
            unit = base = _CPU_CACHELINE
        else:
            unit, base = _GPU_UNIT_STRIDE_ALIGNMENT, _GPU_BASE_ALIGNMENT
        return _resolve(
            tuple(range(len(shape))),
            shape,
            dtype,
            unit_stride_alignment=unit,
            base_alignment=base,
            policy=self,
        )
