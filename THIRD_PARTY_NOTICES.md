# Third-party software used by the CARS artifact (installed separately; not redistributed here)

Licence information was read from the installed distributions' metadata on 15 Nov 2026.

| Package | Version | Licence (as declared) | Role |
|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause | arrays, dense kernels |
| scipy | 1.15.3 | BSD-3-Clause (ARPACK: BSD-style) | sparse matrices, conjugate gradients, ARPACK eigenvalue estimate |
| pypardiso | 0.4.7 | BSD-3-Clause | loads Intel oneMKL PARDISO (called directly through ctypes) |
| mkl | 2026.2.1 | Intel Simplified Software License (proprietary, closed source) | PARDISO sparse Cholesky |
| intel-openmp, intel-cmplr-lib-ur | 2026.2.0 | Intel End User License Agreement for Developer Tools | runtime of MKL |
| tbb, tcmlib | 2023.2.0, 1.6.0 | Intel Simplified Software License | runtime of MKL |
| umf | 1.2.0 | Apache-2.0 with LLVM exceptions | runtime of MKL |
| python-flint | 0.9.0 | MIT (python-flint); bundles FLINT 3.6 (LGPL-3.0-or-later) and GCC runtime libraries (GPL-3.0-or-later with the GCC runtime exception) | exact rational arithmetic and Arb ball arithmetic |
| cholespy | 2.2.0 | BSD-3-Clause (cholespy); the binary bundles SuiteSparse CHOLMOD, whose own licence terms apply (consult the cholespy distribution) | independent Cholesky engine for re-proofs |
| psutil | 7.1.3 | BSD-3-Clause | CPU affinity, priority, memory |
| matplotlib | 3.10.8 | matplotlib licence (PSF-based) | figures |

**Consequences**
- Results that depend on PARDISO depend on a closed-source library. Its conformity with the rounding-error model of Rump's criterion is an assumption of the study. It is cross-checked by CHOLMOD re-proofs and an Arb sample.
- The artifact can be redistributed under the owner's chosen licence without including any of the above binaries. Users install them from PyPI with `environment/requirements-lock.txt`.
