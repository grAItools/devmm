"""Layout contract tests: exhaustive offset injectivity/in-bounds oracle over
every shipped policy, `Aligned` padding postconditions, upper-bound alignment
invariants with provenance, value semantics, `validate()` counterexamples, and
shape edge cases (design §3.6).
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from devmm import (
    Aligned,
    ColMajor,
    Device,
    DeviceOptimal,
    DType,
    Layout,
    LayoutPolicy,
    Permuted,
    RowMajor,
)

CPU = Device.from_string("cpu")
CUDA = Device.from_string("cuda:0")

DEVICES = tuple(
    Device.from_string(s)
    for s in ("cpu", "cuda:0", "cuda:1", "rocm:0", "cuda_host", "rocm_host", "cuda_managed")
)

float32 = DType.from_string("float32")
float64 = DType.from_string("float64")

# Itemsizes 1/4/16 plus 3 bytes: the pad-step computation must also handle an
# itemsize that does not divide the pitch alignment.
DTYPES = (
    DType.from_string("uint8"),
    float32,
    DType.from_string("complex128"),
    DType(0, 24),
)

POLICIES = (RowMajor(), ColMajor(), Permuted((1, 0)), Aligned(RowMajor()), DeviceOptimal())


def all_offsets(layout: Layout, shape: tuple[int, ...]) -> list[int]:
    """Element offset of every index tuple in `shape` (exhaustive)."""
    return [
        sum(i * s for i, s in zip(idx, layout.strides, strict=True))
        for idx in itertools.product(*(range(extent) for extent in shape))
    ]


@st.composite
def cases(draw: st.DrawFn) -> tuple[tuple[int, ...], LayoutPolicy, DType, Device]:
    shape = tuple(draw(st.lists(st.integers(0, 8), min_size=0, max_size=5)))
    perm = tuple(draw(st.permutations(range(len(shape)))))
    policy = draw(
        st.sampled_from(
            [
                RowMajor(),
                ColMajor(),
                Permuted(perm),
                Aligned(RowMajor(), 128, 256),
                Aligned(ColMajor(), 16, 64),
                Aligned(Permuted(perm), 3, 5),
                DeviceOptimal(),
            ]
        )
    )
    return shape, policy, draw(st.sampled_from(DTYPES)), draw(st.sampled_from(DEVICES))


@settings(deadline=None)
@given(case=cases())
def test_offsets_are_distinct_nonnegative_and_in_bounds(
    case: tuple[tuple[int, ...], LayoutPolicy, DType, Device],
) -> None:
    shape, policy, dtype, device = case
    layout = policy(shape, dtype, device)
    offsets = all_offsets(layout, shape)
    assert len(offsets) == math.prod(shape)
    assert len(set(offsets)) == len(offsets)
    if offsets:
        assert min(offsets) >= 0
        assert (max(offsets) + 1) * dtype.itemsize <= layout.required_nbytes
    layout.validate(shape, dtype.itemsize)


@settings(deadline=None)
@given(case=cases())
def test_layout_alignment_bounded_by_policy_and_provenance(
    case: tuple[tuple[int, ...], LayoutPolicy, DType, Device],
) -> None:
    shape, policy, dtype, device = case
    layout = policy(shape, dtype, device)
    assert layout.policy is policy
    assert layout.base_alignment <= policy.base_alignment
    assert layout.required_nbytes % layout.base_alignment == 0
    if len(shape) >= 2:
        # Unit-stride analogue of the upper bound: the innermost padding never
        # reaches the largest pad step the policy may use for this itemsize.
        pitch = layout.strides[layout.permutation[-2]]
        padding = pitch - max(shape[layout.permutation[-1]], 1)
        unit = policy.unit_stride_alignment
        assert 0 <= padding < unit // math.gcd(dtype.itemsize, unit)


@pytest.mark.parametrize(("unit", "base"), [(128, 256), (64, 64), (4, 8), (3, 5)])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: f"itemsize{d.itemsize}")
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 7), (1, 1)])
def test_aligned_pitch_divisible_and_padding_minimal(
    unit: int, base: int, dtype: DType, shape: tuple[int, ...]
) -> None:
    layout = Aligned(RowMajor(), unit, base)(shape, dtype, CPU)
    pitch = layout.strides[-2]
    assert pitch * dtype.itemsize % unit == 0
    minimal = max(shape[-1], 1)
    while minimal * dtype.itemsize % unit:
        minimal += 1
    assert pitch == minimal
    assert layout.required_nbytes % base == 0


def test_unpadded_policies_declare_no_alignment_demands() -> None:
    # No-padding policies report 1 for both upper bounds (§3.6): they never
    # ask anything of the base pointer or the line pitch.
    for policy in (RowMajor(), ColMajor(), Permuted((1, 0))):
        assert policy.base_alignment == 1
        assert policy.unit_stride_alignment == 1


def test_device_optimal_declares_the_gpu_maxima_as_upper_bounds() -> None:
    # Upper-bound semantics (§3.6): a dispatching policy's alignment
    # properties report the maximum it may request for any device — for
    # DeviceOptimal, the GPU values.
    policy = DeviceOptimal()
    assert policy.base_alignment == 256
    assert policy.unit_stride_alignment == 128


def test_device_optimal_dispatches_per_device() -> None:
    host = DeviceOptimal()((3, 5), float32, CPU)
    assert host.base_alignment == 64
    assert host.strides == (16, 1)  # 16 elements * 4 B = one 64-byte line
    assert host.required_nbytes % 64 == 0

    gpu = DeviceOptimal()((3, 5), float32, CUDA)
    assert gpu.base_alignment == 256
    assert gpu.strides == (32, 1)  # 32 elements * 4 B = one 128-byte line
    assert gpu.required_nbytes % 256 == 0


@pytest.mark.parametrize(
    ("device", "base"),
    [
        ("cpu", 64),
        ("cuda_host", 64),
        ("rocm_host", 64),
        ("cuda:0", 256),
        ("rocm:0", 256),
        ("cuda_managed", 256),
    ],
)
def test_device_optimal_treats_host_resident_memory_as_cpu(device: str, base: int) -> None:
    layout = DeviceOptimal()((3, 5), float32, Device.from_string(device))
    assert layout.base_alignment == base


def test_nested_aligned_outer_alignment_wins() -> None:
    layout = Aligned(Aligned(RowMajor(), 64, 64), 128, 256)((3, 5), float32, CPU)
    assert layout.base_alignment == 256
    assert layout.strides == (32, 1)


def test_permuted_orders_strides_by_permutation() -> None:
    # (1, 2, 0): dim 1 outermost, dim 0 innermost.
    layout = Permuted((1, 2, 0))((2, 3, 4), float32, CPU)
    assert layout.permutation == (1, 2, 0)
    assert layout.strides == (1, 8, 2)


# --- value semantics -------------------------------------------------------


def test_layout_policy_is_abstract() -> None:
    with pytest.raises(TypeError):
        LayoutPolicy()  # type: ignore[abstract]


def test_field_backed_policies_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Permuted((1, 0)).permutation = (0, 1)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Aligned(RowMajor()).base_alignment = 512  # type: ignore[misc]


def test_stateless_policies_reject_attribute_injection() -> None:
    # Field-less frozen+slots dataclasses have no writable storage at all;
    # CPython's generated frozen __setattr__ raises TypeError here instead of
    # FrozenInstanceError, but the attribute never lands either way.
    for policy in (RowMajor(), ColMajor(), DeviceOptimal()):
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            policy.base_alignment = 512  # type: ignore[misc]


def test_layout_is_frozen() -> None:
    layout = RowMajor()((2, 2), float32, CPU)
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.required_nbytes = 0  # type: ignore[misc]


def test_policies_are_hashable_dict_keys() -> None:
    table = {policy: repr(policy) for policy in POLICIES}
    assert table[RowMajor()] == repr(RowMajor())
    assert table[Permuted((1, 0))] == repr(Permuted((1, 0)))
    assert table[Aligned(RowMajor())] == repr(Aligned(RowMajor()))
    assert Aligned(RowMajor(), 4, 8) != Aligned(RowMajor(), 4, 16)


def test_layouts_are_hashable_dict_keys() -> None:
    layout = RowMajor()((2, 3), float32, CPU)
    twin = RowMajor()((2, 3), float32, CPU)
    assert twin == layout
    assert hash(twin) == hash(layout)
    assert {layout: "hit"}[twin] == "hit"


# --- validate() ------------------------------------------------------------


def test_validate_accepts_hand_built_padded_f_order() -> None:
    layout = Layout(permutation=(1, 0), strides=(1, 4), required_nbytes=64, base_alignment=1)
    layout.validate((3, 4), itemsize=4)
    assert layout.policy is None


@pytest.mark.parametrize(
    ("layout", "shape", "itemsize"),
    [
        # Overlapping: pitch 1 < extent 2 of the inner dim.
        (Layout((0, 1), (1, 1), 16, 1), (2, 2), 1),
        # Overlapping: pitch 3 < row length 4.
        (Layout((0, 1), (3, 1), 64, 1), (2, 4), 1),
        # Zero stride (broadcast alias).
        (Layout((0, 1), (0, 1), 16, 1), (2, 2), 1),
        (Layout((0, 1), (4, 0), 16, 1), (2, 2), 1),
        # Negative stride.
        (Layout((0, 1), (-4, 1), 64, 1), (2, 4), 1),
        # Non-unit innermost stride is not permutation+padding derivable.
        (Layout((0, 1), (4, 2), 16, 1), (2, 2), 1),
        # Stride chain not an exact multiple between adjacent dims.
        (Layout((0, 1, 2), (7, 3, 1), 64, 1), (2, 2, 2), 1),
        # Permutation is not a permutation.
        (Layout((0, 0), (2, 1), 16, 1), (2, 2), 1),
        # Rank mismatch.
        (Layout((0, 1), (2, 1), 16, 1), (2, 2, 2), 1),
        # Out of bounds: last element ends at byte 32 > 31.
        (Layout((0, 1), (4, 1), 31, 1), (2, 4), 4),
        # Non-integer byte size.
        (Layout((0, 1), (4, 1), 32.0, 1), (2, 4), 4),  # type: ignore[arg-type]
        # Negative extent.
        (Layout((0, 1), (4, 1), 32, 1), (2, -4), 4),
        # Non-positive base alignment.
        (Layout((0, 1), (4, 1), 32, 0), (2, 4), 4),
        # Non-positive / non-integer itemsize.
        (Layout((0, 1), (4, 1), 32, 1), (2, 4), 0),
        (Layout((0, 1), (4, 1), 32, 1), (2, 4), 4.0),
    ],
)
def test_validate_rejects(layout: Layout, shape: tuple[int, ...], itemsize: int) -> None:
    with pytest.raises(ValueError):
        layout.validate(shape, itemsize)


# --- is_contiguous ---------------------------------------------------------


def test_policy_layouts_are_dense_over_their_envelope() -> None:
    assert RowMajor()((2, 3), float32, CPU).is_contiguous
    assert ColMajor()((2, 3), float32, CPU).is_contiguous


def test_padded_layouts_are_dense_over_their_padded_envelope() -> None:
    # `Layout` stores no shape, so contiguity is relative to the padded
    # extents implied by the strides (see the property's docstring).
    layout = Aligned(RowMajor(), 128, 256)((4, 100), float32, CPU)
    assert layout.strides == (128, 1)
    assert layout.is_contiguous


def test_hand_built_gapped_layouts_are_not_contiguous() -> None:
    assert not Layout((0, 1), (4, 2), 32, 1).is_contiguous  # non-unit innermost
    assert not Layout((0, 1, 2), (7, 3, 1), 64, 1).is_contiguous  # ragged chain
    assert not Layout((0, 1), (1, 4), 32, 1).is_contiguous  # permutation disagrees


# --- construction errors ---------------------------------------------------


def test_permuted_rejects_non_permutations() -> None:
    with pytest.raises(ValueError):
        Permuted((0, 0))
    with pytest.raises(ValueError):
        Permuted((0, 2))


def test_permuted_rejects_rank_mismatch() -> None:
    with pytest.raises(ValueError):
        Permuted((1, 0))((2, 3, 4), float32, CPU)


def test_policies_reject_negative_extents() -> None:
    with pytest.raises(ValueError):
        RowMajor()((2, -1), float32, CPU)


def test_aligned_rejects_non_positive_alignments() -> None:
    with pytest.raises(ValueError):
        Aligned(RowMajor(), 0, 256)
    with pytest.raises(ValueError):
        Aligned(RowMajor(), 128, -256)


# --- edge cases ------------------------------------------------------------


def test_scalar_layout_has_empty_strides_and_one_element() -> None:
    layout = RowMajor()((), float32, CPU)
    assert layout.permutation == ()
    assert layout.strides == ()
    assert layout.required_nbytes == float32.itemsize
    assert layout.is_contiguous
    layout.validate((), float32.itemsize)


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), (0, 0)])
def test_zero_extent_shapes_need_no_bytes(shape: tuple[int, ...]) -> None:
    for policy in (RowMajor(), Aligned(RowMajor(), 128, 256)):
        layout = policy(shape, float32, CPU)
        assert layout.required_nbytes == 0
        assert all(s >= 1 for s in layout.strides)
        layout.validate(shape, float32.itemsize)


def test_extent_one_dims_keep_cumulative_strides() -> None:
    layout = RowMajor()((3, 1, 5), float32, CPU)
    assert layout.strides == (5, 5, 1)


def test_huge_extents_stay_exact_ints() -> None:
    layout = RowMajor()((1 << 20, 1 << 20, 1 << 20), float64, CPU)
    assert layout.required_nbytes == (1 << 60) * 8
    assert type(layout.required_nbytes) is int
    assert all(type(s) is int for s in layout.strides)
    assert layout.strides == (1 << 40, 1 << 20, 1)
