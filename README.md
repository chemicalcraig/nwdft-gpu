# GPU Acceleration of Real-Time TDDFT in NWChem
## Version 7.3.1 (<https://github.com/nwchemgit/nwchem/archive/refs/tags/v7.3.1-release.tar.gz>)
N.B. Built specifically for Ampere architecture. May work on others. Haven't tested it yet.
To use: replace $NWCHEM_TOP/src/nwdft with the enclosed repository, compile with CUDA flags (for example):
```shell
export TCE_CUDA=Y
export CUDA_LIBS="-L/usr/local/cuda-12.4/lib64/ -lcudart -lcublas"
export CUDA_INCLUDE="-I/usr/local/cuda-12.4/include/"
```

## 1. Abstract
We present a high-performance GPU acceleration framework for the Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) module within NWChem. This work addresses the critical bottleneck of Fock matrix construction by offloading the computationally intensive Coulomb ($J$) and Exact Exchange ($K$) integral evaluations to NVIDIA GPUs. The implementation features a hybrid CPU/GPU architecture, a specialized J/K-Engine kernel based on the Obara-Saika recurrence relation, and optimizations targeting modern Ampere architectures.

## 2. Introduction
RT-TDDFT is a powerful method for simulating electron dynamics, but its $O(N^4)$ scaling with system size limits its applicability to large systems. The propagation of the density matrix $P(t)$ requires rebuilding the Fock matrix $F(t)$ at every time step:
$$ F_{\mu\nu}(t) = H^{core}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma}(t) [ (\mu\nu|\lambda\sigma) - \gamma (\mu\lambda|\nu\sigma) ] + V^{XC}_{\mu\nu} $$
where $\gamma$ is the exact exchange fraction. The two-electron repulsion integrals $(\mu\nu|\lambda\sigma)$ dominate the cost. We introduce a GPU-accelerated engine to compute these terms, replacing the legacy CPU-based `fock_2e` routine.

## 3. Methodology & Architecture

### 3.1. Hybrid Architecture
The implementation follows a "Rank 0 Offloading" model, designed for workstations and single-node multi-GPU clusters.
*   **Host (CPU):** Manages the main RT-TDDFT propagation loop, computes $H^{core}$ and grid-based $V^{XC}$, and handles MPI communication.
*   **Device (GPU):** Asynchronously computes the Coulomb ($J$) and Exchange ($K$) matrices. Data transfer is minimized by uploading the basis set once and streaming only the time-dependent density matrix $P(t)$.

### 3.2. Data Structures
To facilitate efficient GPU access, the standard NWChem basis set (linked lists) is flattened into a structure-of-arrays (SoA) format:
```cpp
struct Shell {
    double x, y, z; // Center coordinates
    int L;          // Angular momentum (0=S, 1=P, ...)
    int nprim;      // Number of primitives
    int ptr_exp;    // Pointer to exponents
    int ptr_coef;   // Pointer to contraction coefficients
    int basis_idx;  // Global index offset
};
```
This data is uploaded to the GPU during initialization via `rttddft_gpu_upload_basis`.

## 4. The J/K-Engine Kernel

### 4.1. Integral Evaluation
We implemented a specialized CUDA kernel based on the **Obara-Saika (OS) recurrence relations**. This choice allows for efficient on-the-fly evaluation of integrals for S and P orbitals, covering the majority of organic chemistry applications (e.g., STO-3G, 6-31G).

**Base Integrals $(ss|ss)^{(m)}$:**
The base integrals are computed using the Boys function $F_m(T)$. We use a hybrid evaluation strategy:
*   **Small $T$:** Taylor expansion for numerical stability.
*   **Large $T$:** Asymptotic approximation using `erf` and downward recurrence.

**Recurrence Relations:**
Higher angular momentum integrals are generated via:
1.  **Vertical Recurrence (VRR) on Bra (A):** Generates $(ps|ss)$ and $(ds|ss)$ from $(ss|ss)^{(m)}$.
2.  **Horizontal Recurrence (HRR) on Bra (A):** Transfers momentum to center B, generating $(sp|ss)$ and $(pp|ss)$.
3.  **VRR/HRR on Ket (C):** Generates $(ab|ps)$, $(ab|sp)$, and $(ab|pp)$ by treating the Bra pair $(ab)$ as a compound source.

### 4.2. Simultaneous J/K Construction
To support hybrid functionals (e.g., B3LYP, PBE0), the kernel computes $J$ and $K$ matrices simultaneously. This provides a 2x theoretical speedup over separate passes by reusing the expensive ERI evaluation.
*   **Coulomb:** $J_{\mu\nu} 	+= 	\sum_{\lambda\sigma} P_{\lambda\sigma} (\mu\nu|\lambda\sigma)$
*   **Exchange:** $K_{\mu\lambda} 	+= 	 P_{\nu\sigma} (\mu\nu|\lambda\sigma)$ (and symmetric permutations)

### 4.3. Parallelization Strategy
The kernel employs a massive threading strategy to hide memory latency:
*   **Grid:** Iterates over Bra shell pairs $(i, j)$.
*   **Threads (128/block):** Parallelize the loop over Ket shell pairs $(k, l)$ via grid-striding.
*   **Accumulation:** 
    *   **$J$ Matrix:** Uses thread-local register accumulation followed by a single atomic reduction per block to global memory.
    *   **$K$ Matrix:** Uses `atomicAdd_safe` (robust double-precision atomics) to scatter contributions to the global matrix.

## 5. Optimization & Performance

### 5.1. Atomic Safety
We implemented a robust `atomicAdd_safe` wrapper. It utilizes native hardware atomics on supported architectures (Pascal `sm_60` and newer, including Ampere `sm_80`) and falls back to a CAS (Compare-And-Swap) loop on older hardware, ensuring portability.

### 5.2. Register Pressure
By unrolling the OS recurrence loops and accumulating $J$ contributions in registers (`acc_J[3][3]`), we significantly reduced global memory traffic.

### 6. Propagator Acceleration
The core of the RT-TDDFT propagation involves solving the Liouville-von Neumann equation. In the Magnus expansion formalism (specifically the second-order Magnus propagator), the time-evolution operator  is approximated as an exponential of the Fock matrix . The update of the density matrix  is computed using the Baker-Campbell-Hausdorff (BCH) expansion:

 
 

This iterative expansion requires repeated dense matrix multiplications () and matrix additions (). For a system with  basis functions, each step scales as . We identified this dense linear algebra loop as the primary candidate for GPU offloading.

6.1. Implementation Details
The implementation follows a hybrid Fortran/C/CUDA architecture:

CUDA/C Wrapper Layer: A C-based wrapper (rttddft_gpu.cu) interfaces directly with the NVIDIA cuBLAS library. It manages GPU memory allocation (cudaMalloc), data transfer (cudaMemcpy), and linear algebra operations (cublasZgemm, cublasZaxpy).
Fortran Interface: A Fortran module (rttddft_gpu.fh) exposes these C functions to the legacy Fortran codebase using standard ISO C bindings or direct external linking, ensuring compatibility with NWChem's memory management.
Propagator Refactoring: The core propagator routine prop_magnus_exp_bch.F was refactored. A conditional execution path was introduced:
Rank 0 Offloading: The MPI rank 0 process takes ownership of the GPU. It allocates device memory and transfers the density () and Fock () matrices to the device.
Device-Resident Loop: The iterative BCH expansion loop executes entirely on the GPU. Intermediate commutators are stored in device memory, eliminating costly PCIe transfers during the iteration.
Synchronization: Upon convergence, the updated density matrix is copied back to the host and broadcast to all MPI ranks via Global Arrays (ga_brdcst).

## 7. Validation
The implementation was validated using:
*   **H2 / STO-3G:** Validated (ss|ss) integrals and J-matrix construction.
*   **CH4 / 6-31G:** Validated (sp|sp), (pp|ss), etc., ensuring correct P-orbital support.
*   **Hybrid DFT:** Verified correct scaling and accumulation of the Exchange matrix against CPU references.

## 8. Future Work
*   **D/F Orbitals:** Extending the OS engine to support D and F functions.
*   **Shared Memory Tiling:** Implementing shared memory caching for the density matrix $P$ to further reduce global memory bandwidth pressure.
*   **Multi-GPU:** Distributing the shell quartet workload across multiple GPUs using NCCL.
