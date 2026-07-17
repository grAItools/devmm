"""Provider-direction "install devmm into X" integrations (design §6).

Every module here mutates third-party global state through an explicit
`install()` call — never as a side effect of import or of
`set_current_memory_resource` — and returns an `Installer` whose
`uninstall()`/context-manager form restores the prior state. Composing an
install with the matching consume-direction MR (e.g.
`CupyAllocatorMemoryResource` under `integrations.cupy.install`) is a
direct allocation cycle and raises.
"""

from devmm.integrations import cupy, numba, numpy, rmm
from devmm.integrations._support import Installer

__all__ = ["Installer", "cupy", "numba", "numpy", "rmm"]
