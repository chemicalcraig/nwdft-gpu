# RT-TDDFT GPU Acceleration Report

This report summarizes the implementation and verification of GPU acceleration for the Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) module in NWChem.

## 1. Executive Summary

We have successfully implemented a hybrid CPU-GPU acceleration strategy for RT-TDDFT. The solution focuses on two key bottlenecks:
1.  **Time Propagation:** Offloading the matrix exponential ($e^{-iF\Delta t}$) to the GPU using `cuBLAS`.
2.  **Fock Matrix Construction:** Implementing a Resolution of Identity (RI-J) Coulomb engine that pre-computes 3-center integrals on the CPU and performs fast tensor contractions on the GPU.

This approach bypasses the complexity of implementing 4-center electron repulsion integrals (ERIs) on the GPU while delivering significant performance improvements for Coulomb-dominated calculations.

---

## 2. Implementation Details

### 2.1 GPU Propagator (Matrix Exponentials)
*   **Kernel:** `rttddft_gpu_propagate_pseries_` (CUDA)
*   **Algorithm:** Adaptive Taylor series expansion with scaling and squaring.
*   **Performance:** Fully offloaded to `cuBLAS` (Zgemm, Zaxpy).
*   **Status:** Converges to $10^{-12}$ tolerance. Verified against CPU reference.

### 2.2 RI-J Acceleration (Density Fitting)
*   **Strategy:**
    1.  **Initialization (CPU):** Compute 2-center inverse metric $(P|Q)^{-1}$ and 3-center integrals $(P|\mu\nu)$ using NWChem's optimized CPU integral engines (`int_2e2c`, `int_2e3c`).
    2.  **Transfer:** Upload tensors to GPU global memory (`rttddft_gpu_upload_ri`).
    3.  **Contraction (GPU):** At each time step, construct the Coulomb matrix $J$ via fast matrix-vector multiplications:
        $$ V_Q = \sum_{\mu\nu} (Q|\mu\nu) P_{\mu\nu} \quad (\text{Gemv}) $$
        $$ C_P = \sum_Q (P|Q)^{-1} V_Q \quad (\text{Gemv}) $$
        $$ J_{\mu\nu} = \sum_P (P|\mu\nu) C_P \quad (\text{Gemv}) $$
*   **Basis Support:** Fully supports high angular momentum shells (F, G, etc.) via the CPU integral engine.
*   **Status:** Functional and verified with `cc-pVDZ-RI` fitting basis.

### 2.3 Legacy Support & Bug Fixes
*   **F-Shell Crash:** Fixed a critical bug in the legacy GPU kernel that caused crashes/NaNs when encountering F-shells (L=3). The kernel now safely handles higher angular momenta.
*   **Basis Parsing:** Updated the Fortran basis parser to correctly handle 64-bit integer passing between Fortran and C/CUDA, resolving a "primitive count mismatch" error.

---


## 3. Usage Guide

To utilize the new GPU features, add the following to your NWChem input file:

1.  **Enable GPU Compilation:** Ensure `RTTDDFT_CUDA=Y` is set during compilation.
2.  **Define Fitting Basis:**
    ```
    basis "cd basis"
      * library "cc-pvdz-ri"
    end
    ```
    *Note: If "cd basis" is omitted, the code falls back to the slower/legacy path.*
3.  **Run:**
    ```
    rt_tddft
      ...
      exp pseries  # Activates GPU Power Series
    end
    ```

## 5. Future Work
*   **Exchange Acceleration:** Implement RI-K or semi-numerical exchange to accelerate hybrid functionals.
*   **Multi-GPU Support:** Distribute the RI tensors across multiple GPUs for larger systems.
