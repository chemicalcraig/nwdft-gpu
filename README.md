# RT-TDDFT GPU Acceleration: Master Progress & Implementation Details

## 1. Executive Summary (Current Status)
We have implemented a hybrid CPU/GPU acceleration framework for the Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) module in NWChem. The current stable implementation accelerates two critical bottlenecks:
1.  **Time Propagation:** The matrix exponential ($e^{-iF\Delta t}$) and density matrix updates are fully offloaded to the GPU using `cuBLAS`.
2.  **Coulomb Matrix ($J$) Construction:** We utilize the Resolution of Identity (RI) approximation (Density Fitting) to convert expensive 4-center integrals into efficient GPU tensor contractions. This approach supports high angular momentum basis functions (S, P, D, F, G, etc.).

## 2. Usage Guide

### 2.1 Compilation
To enable GPU acceleration, configure NWChem with `RTTDDFT_CUDA=Y` and provide paths to the CUDA Toolkit.

```bash
export RTTDDFT_CUDA=Y
export CUDA_LIBS="-L/usr/local/cuda/lib64 -lcudart -lcublas"
export CUDA_INCLUDE="-I/usr/local/cuda/include"
cd $NWCHEM_TOP/src
make nwchem_config
make
```

### 2.2 Input Specifications
To activate the GPU features, specify a fitting basis for RI-J and select the `pseries` propagator.

```nwchem
basis "cd basis"
  * library "cc-pvdz-ri"
end

rt_tddft
  tmax 10.0
  dt 0.1
  exp pseries  # Activates GPU-accelerated matrix exponential
end
```
*Note: Omitting `cd basis` reverts to the legacy CPU or direct-GPU path.*

## 3. Technical Implementation & Theoretical Background

### 3.1 GPU Propagator (Magnus Expansion)
The core of the RT-TDDFT propagation involves solving the Liouville-von Neumann equation. In the second-order Magnus propagator, the time-evolution operator $U(t)$ is approximated as an exponential of the Fock matrix $F(t)$.

**BCH Expansion:**
The update of the density matrix $P(t)$ is computed using the Baker-Campbell-Hausdorff (BCH) expansion:
$$ P(t+\Delta t) = e^{-i F \Delta t} P(t) e^{i F \Delta t} = P(t) + \frac{1}{1!} [-iF\Delta t, P(t)] + \frac{1}{2!} [-iF\Delta t, [-iF\Delta t, P(t)]] + \dots $$

**Implementation:**
*   **Offloading:** The MPI rank 0 process takes ownership of the GPU, allocating device memory and transferring $P$ and $F$.
*   **Computation:** The iterative BCH expansion and "scaling and squaring" for the matrix exponential are performed entirely on the GPU using `cuBLAS` (`ZGEMM`, `ZAXPY`). This avoids expensive PCIe transfers during the inner loops.

### 3.2 Coulomb Matrix: RI-J Acceleration (Current)
To support high angular momentum shells effectively, we shifted from a direct GPU kernel to an RI-J approach.
$$ (\mu\nu|\lambda\sigma) \approx \sum_{PQ} (\mu\nu|P) (P|Q)^{-1} (Q|\lambda\sigma) $$

*   **Pre-computation (CPU):** 3-center integrals $(\mu\nu|P)$ and the inverse metric $(P|Q)^{-1}$ are computed using NWChem's optimized CPU engines.
*   **Contraction (GPU):** The $J$ matrix is constructed via fast `GEMV` operations.
    $$ V_Q = \sum_{\mu\nu} (Q|\mu\nu) P_{\mu\nu} $$
    $$ J_{\mu\nu} = \sum_P (P|\mu\nu) C_P \quad \text{where} \quad C_P = \sum_Q (P|Q)^{-1} V_Q $$

## 4. Previous Developments: Direct J/K Kernel
*This section details the direct GPU kernel implementation (Obara-Saika). While functional for S/P orbitals, it is currently superseded by RI-J for general use cases involving D/F shells.*

### 4.1 Methodology
We implemented a specialized CUDA kernel based on the **Obara-Saika (OS) recurrence relations** to compute electron repulsion integrals (ERIs) on the fly.
*   **Base Integrals:** Computed via Boys function $F_m(T)$ using Taylor expansion (small T) or asymptotic approximation (large T).
*   **Recurrence:** Vertical (VRR) and Horizontal (HRR) recurrences generate higher angular momentum integrals from base $(ss|ss)^{(m)}$ integrals.

### 4.2 Architecture
*   **Rank 0 Offloading:** Similar to the propagator, Rank 0 manages the GPU.
*   **Structure-of-Arrays (SoA):** The basis set is flattened into SoA format for coalesced GPU access.
*   **Simultaneous J/K:** The kernel computes Coulomb ($J$) and Exchange ($K$) matrices simultaneously to reuse ERI evaluations.
    *   $J_{\mu\nu} += \sum_{\lambda\sigma} P_{\lambda\sigma} (\mu\nu|\lambda\sigma)$
    *   $K_{\mu\lambda} += P_{\nu\sigma} (\mu\nu|\lambda\sigma)$

### 4.3 Optimizations
*   **Atomic Safety:** Implemented `atomicAdd_safe` wrapper supporting native double-precision atomics (Pascal+) and CAS fallbacks.
*   **Register Tiling:** Heavy use of registers (`acc_J[3][3]`) to reduce global memory traffic.

## 5. Validation History
*   **H2 / STO-3G:** Verified direct Kernel (ss|ss) integrals.
*   **CH4 / 6-31G:** Verified direct Kernel for P-orbitals.
*   **RI-J (Current):** Verified with `cc-pVDZ-RI` fitting basis; supports D/F shells correctly.
*   **Propagator:** Converges to $10^{-12}$ tolerance against CPU reference.

## 6. Future Work
*   **Exchange Acceleration:** Implement RI-K or semi-numerical exchange to accelerate hybrid functionals (currently the main bottleneck).
*   **Multi-GPU:** Distribute the RI tensors and shell quartets across multiple GPUs using NCCL.
*   **Shared Memory Tiling:** Further optimize the density matrix access patterns.
