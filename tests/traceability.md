# Traceability — contracts to tests

Maps enforced contracts to the tests that pin them. One row per contract;
extend this table as phases land (see `work/devmm-implementation-plan.md`).

| Contract | Source | Test |
| --- | --- | --- |
| Package imports and exposes `__version__` | design §2 | `tests/test_smoke.py::test_package_imports` |
| Public API surface is frozen by snapshot (`DType`, `Device`, `DeviceType`, `Layout`, `LayoutPolicy`, shipped policies) | plan p00/p01/p02, Architecture decisions | `tests/test_public_api.py::test_all_matches_snapshot` |
| Exported member signatures match the snapshot | plan p00, Architecture decisions | `tests/test_public_api.py::test_exported_member_signatures_match_snapshot` |
| Every dtype alias carries the exact `dlpack.h` `(code, bits, lanes)` triple | design §3.7 | `tests/test_dtypes.py::test_alias_matches_dlpack_triple` |
| Every dtype alias reports the right itemsize | design §3.7 | `tests/test_dtypes.py::test_alias_itemsize` |
| `DType.from_string` accepts Array-API dtype strings, rejects everything else | design §3.7 | `tests/test_dtypes.py::test_from_string_returns_alias`, `tests/test_dtypes.py::test_from_string_rejects_unknown` |
| `DType.from_any` duck-types NumPy dtypes via `.kind`/`.itemsize`; unsupported kinds/objects raise | design §3.7 | `tests/test_dtypes.py::test_from_any_duck_typed_numpy_like`, `tests/test_dtypes.py::test_from_any_rejects_unsupported_kind`, `tests/test_dtypes.py::test_from_any_rejects_non_dtype_objects` |
| `DType` is frozen and hashes/compares by value | docs/style.md, value objects | `tests/test_dtypes.py::test_dtype_is_frozen`, `tests/test_dtypes.py::test_dtype_is_hashable_and_equal_by_value` |
| NumPy differential: itemsize equality and `np.dtype` round-trip for every alias with a counterpart | design §3.7, §9 | `tests/test_dtypes_numpy.py::test_itemsize_matches_numpy`, `tests/test_dtypes_numpy.py::test_numpy_dtype_round_trips_to_alias` |
| `DeviceType` values are the exact DLPack `DLDeviceType` codes | design §3.1 | `tests/test_device.py::test_device_type_codes_match_dlpack` |
| `Device.from_string` parses well-formed strings; bare name means index 0 | design §3.1 | `tests/test_device.py::test_from_string_parses_valid`, `tests/test_device.py::test_from_string_bare_name_defaults_to_index_zero` |
| `Device.from_string` rejects malformed strings with `ValueError` | design §3.1 | `tests/test_device.py::test_from_string_rejects_malformed`, `tests/test_device.py::test_from_string_rejects_malformed_examples` |
| `Device` round-trips through `str()`; frozen, hashable, equal by value | design §3.1; docs/style.md | `tests/test_device.py::test_from_string_format_round_trip`, `tests/test_device.py::test_device_hash_and_equality`, `tests/test_device.py::test_device_is_frozen` |
| `Device.__dlpack_device__() == (int(type), index)` | design §3.1 | `tests/test_device.py::test_dlpack_device_is_code_index_pair` |
| Importing devmm core never pulls `numpy` into `sys.modules` | design §3.7; docs/style.md | `tests/test_import_hygiene.py::test_core_import_does_not_import_numpy` |
| Zero runtime dependencies: wheel installs and imports in a bare venv | design §8 | `tests/test_packaging.py::test_wheel_imports_in_bare_venv` |
| Wheel metadata declares no unconditional `Requires-Dist` | design §8 | `tests/test_packaging.py::test_wheel_declares_no_runtime_dependencies` |
| Wheel ships only `devmm/` (incl. `py.typed`), never `tests/` | plan p00, Risks | `tests/test_packaging.py::test_wheel_packages_only_devmm` |
| Every shipped policy yields distinct, non-negative, in-bounds element offsets (exhaustive, ndim<=5) | design §3.6 | `tests/test_layout.py::test_offsets_are_distinct_nonnegative_and_in_bounds` |
| Row-/column-major element strides and `required_nbytes` match NumPy exactly | design §3.6, §9 | `tests/test_layout_numpy.py::test_row_major_matches_numpy`, `tests/test_layout_numpy.py::test_col_major_matches_numpy` |
| `Aligned` line pitch is `unit_stride_alignment`-divisible with minimal padding; `required_nbytes % base_alignment == 0` | design §3.6 | `tests/test_layout.py::test_aligned_pitch_divisible_and_padding_minimal` |
| `layout.base_alignment <= policy.base_alignment` (and pitch-padding analogue); `layout.policy is policy` | design §3.6 | `tests/test_layout.py::test_layout_alignment_bounded_by_policy_and_provenance` |
| `DeviceOptimal` dispatches host- vs GPU-resident alignments within its declared bounds | design §3.6 | `tests/test_layout.py::test_device_optimal_dispatches_per_device`, `tests/test_layout.py::test_device_optimal_treats_host_resident_memory_as_cpu` |
| Policies and layouts are frozen, hashable dict keys, equal by value | design §3.6; docs/style.md | `tests/test_layout.py::test_field_backed_policies_are_frozen`, `tests/test_layout.py::test_stateless_policies_reject_attribute_injection`, `tests/test_layout.py::test_layout_is_frozen`, `tests/test_layout.py::test_policies_are_hashable_dict_keys`, `tests/test_layout.py::test_layouts_are_hashable_dict_keys` |
| `Layout.validate()` rejects overlapping/negative/zero/non-derivable strides, rank mismatches, undersized or non-integer `required_nbytes`; accepts sound hand-built layouts | design §3.6 | `tests/test_layout.py::test_validate_rejects`, `tests/test_layout.py::test_validate_accepts_hand_built_padded_f_order` |
| `is_contiguous` reports dense offset envelopes; gapped hand-built strides are not contiguous | design §3.6 | `tests/test_layout.py::test_policy_layouts_are_dense_over_their_envelope`, `tests/test_layout.py::test_padded_layouts_are_dense_over_their_padded_envelope`, `tests/test_layout.py::test_hand_built_gapped_layouts_are_not_contiguous` |
| Layout edge cases: ndim=0, zero/one extents, huge extents with exact integer-only `required_nbytes`/strides | design §3.6 | `tests/test_layout.py::test_scalar_layout_has_empty_strides_and_one_element`, `tests/test_layout.py::test_zero_extent_shapes_need_no_bytes`, `tests/test_layout.py::test_extent_one_dims_keep_cumulative_strides`, `tests/test_layout.py::test_huge_extents_stay_exact_ints` |
| Policies reject malformed inputs (negative extents, non-permutations, rank mismatch, non-positive alignments) | design §3.6 | `tests/test_layout.py::test_policies_reject_negative_extents`, `tests/test_layout.py::test_permuted_rejects_non_permutations`, `tests/test_layout.py::test_permuted_rejects_rank_mismatch`, `tests/test_layout.py::test_aligned_rejects_non_positive_alignments` |
| `gpu_cuda`/`gpu_rocm` tests skip unless `DEVMM_GPU` opts in | design §9 | enforced by `tests/conftest.py` (hook); no pinning test yet |
| `recording_mr` fixture placeholder skips until `devmm.testing` provides it | design §9 | enforced by `tests/conftest.py` (fixture); no pinning test yet |
