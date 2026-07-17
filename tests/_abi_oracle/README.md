# DLPack ABI oracle

Compiled reference for the ctypes mirrors in `devmm/_dlpack/_abi.py`
(design §7.1); exercised by `tests/test_dlpack_abi.py`.

- `dlpack.h` — vendored verbatim from the DLPack **v1.1** release tag
  (`dmlc/dlpack`, `include/dlpack/dlpack.h`,
  sha256 `2540410479f23d62d34c02cb5fce54e4cf165fb315033e9cd948452a65887208`).
  Bumping the pin means updating `DLPACK_VERSION` in `_abi.py` and
  regenerating the snapshots.
- `abi_oracle.c` — prints `sizeof`/`offsetof` of every DLPack struct as
  JSON. On machines with a C compiler (T1) the test suite compiles and
  runs it, diffing the real compiler's layout against ctypes.
- `snapshots/<platform>.json` — the oracle's output, committed for each
  target platform so compiler-less machines (T0) still verify the layout.
  All four targets are 64-bit ABIs (8-byte pointers, 4-byte `int`/enums),
  so the snapshots currently coincide byte-for-byte.

Regenerating a snapshot (run on the target platform):

```sh
cc -std=c11 abi_oracle.c -o abi_oracle
./abi_oracle > snapshots/<platform>.json
```
