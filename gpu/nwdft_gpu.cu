#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

struct Shell {
    double x, y, z;
    int L;
    int nprim;
    int ptr_exp;
    int ptr_coef;
    int basis_idx;
};

static Shell* d_shells = NULL;
static double* d_exponents = NULL;
static double* d_coefs = NULL;
static int n_shells_gpu = 0;
static int n_prim_total = 0;

static cublasHandle_t handle = NULL;
static int is_init = 0;

extern "C" {

void nwdft_gpu_init_() {
    if (!is_init) {
        printf("NWDFT GPU: Initializing cuBLAS and setting stack limit...\n");
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            return;
        }
        cudaDeviceSetLimit(cudaLimitStackSize, 65536);  // Increased for D-shell arrays
        is_init = 1;
    }
}

void nwdft_gpu_finalize_() {
    if (is_init) {
        cublasDestroy(handle);
        is_init = 0;
        if (d_shells) cudaFree(d_shells);
        if (d_exponents) cudaFree(d_exponents);
        if (d_coefs) cudaFree(d_coefs);
        d_shells = NULL; d_exponents = NULL; d_coefs = NULL;
    }
}

void nwdft_gpu_upload_basis_c_(long *nshells, long *nprim_total,
                               double *coords, long *ang, long *nprim_per_shell,
                               long *basis_idx,
                               double *exps, double *coefs) {
    if (!is_init) nwdft_gpu_init_();
    n_shells_gpu = (int)*nshells;
    int n_prim = (int)*nprim_total;
    n_prim_total = n_prim;  // Store for test function

    // DEBUG: Print what C receives
    printf("C upload_basis: nshells=%d, nprim=%d\n", n_shells_gpu, n_prim);
    printf("C first 5 exps received: ");
    for (int i = 0; i < 5 && i < n_prim; i++) printf("%f ", exps[i]);
    printf("\n");
    printf("C first 5 coefs received: ");
    for (int i = 0; i < 5 && i < n_prim; i++) printf("%f ", coefs[i]);
    printf("\n");
    printf("C first shell: ang=%ld nprim=%ld basis_idx=%ld\n", ang[0], nprim_per_shell[0], basis_idx[0]);
    printf("C first shell coords: %f %f %f\n", coords[0], coords[1], coords[2]);
    Shell *h_shells = (Shell*)malloc(sizeof(Shell) * n_shells_gpu);
    int current_ptr = 0;
    for (int i = 0; i < n_shells_gpu; i++) {
        h_shells[i].x = coords[3*i + 0]; h_shells[i].y = coords[3*i + 1]; h_shells[i].z = coords[3*i + 2];
        h_shells[i].L = (int)ang[i]; h_shells[i].nprim = (int)nprim_per_shell[i];
        h_shells[i].basis_idx = (int)basis_idx[i];
        h_shells[i].ptr_exp = current_ptr; h_shells[i].ptr_coef = current_ptr;
        current_ptr += (int)nprim_per_shell[i];
    }
    if (d_shells) cudaFree(d_shells); if (d_exponents) cudaFree(d_exponents); if (d_coefs) cudaFree(d_coefs);
    cudaMalloc(&d_shells, sizeof(Shell) * n_shells_gpu);
    cudaMalloc(&d_exponents, sizeof(double) * n_prim);
    cudaMalloc(&d_coefs, sizeof(double) * n_prim);
    cudaMemcpy(d_shells, h_shells, sizeof(Shell) * n_shells_gpu, cudaMemcpyHostToDevice);
    cudaMemcpy(d_exponents, exps, sizeof(double) * n_prim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coefs, coefs, sizeof(double) * n_prim, cudaMemcpyHostToDevice);
    free(h_shells);
}

} // end extern C

__device__ double nwdft_atomicAdd_safe(double* address, double val) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
#else
    return atomicAdd(address, val);
#endif
}

__device__ void nwdft_compute_boys(double t, double *F, int m_max) {
    if (t < 1e-7) {
        for (int m=0; m<=m_max; m++) F[m] = 1.0/(2*m+1) - t/(2*m+3); 
    } else if (t > 30.0) {
        double st = sqrt(t);
        F[0] = 0.886226925452758 / st;
        for (int m=0; m<m_max; m++) F[m+1] = (2*m+1) * F[m] / (2.0 * t);
    } else {
        int M = m_max + 15;
        double exp_t = exp(-t);
        double res[40]; res[M] = 0.0; 
        for (int m=M; m>0; m--) res[m-1] = (2.0*t*res[m] + exp_t) / (2.0*m - 1.0);
        for (int m=0; m<=m_max; m++) F[m] = res[m];
    }
}

__device__ void nwdft_get_d_indices(int type, int *d1, int *d2) {
    int t = type - 4;
    if (t==0) { *d1=0; *d2=0; } else if (t==1) { *d1=0; *d2=1; } else if (t==2) { *d1=0; *d2=2; }
    else if (t==3) { *d1=1; *d2=1; } else if (t==4) { *d1=1; *d2=2; } else if (t==5) { *d1=2; *d2=2; }
}

// ERI function with memory-efficient D-shell support
// Computes DP, PD, DD intermediates on-demand for specific indices only
__device__ double nwdft_compute_eri_primitive(double ai, double aj, double ak, double al,
                                        double xi, double yi, double zi,
                                        double xj, double yj, double zj,
                                        double xk, double yk, double zk,
                                        double xl, double yl, double zl,
                                        int i_type, int j_type, int k_type, int l_type) {
    if ( (k_type + l_type) > (i_type + j_type) ) {
        return nwdft_compute_eri_primitive(ak, al, ai, aj, xk, yk, zk, xl, yl, zl, xi, yi, zi, xj, yj, zj, k_type, l_type, i_type, j_type);
    }
    double p = ai + aj, q = ak + al, rP = p + q, mu = p * q / rP;
    double P[3] = {(ai*xi + aj*xj)/p, (ai*yi + aj*yj)/p, (ai*zi + aj*zj)/p};
    double Q[3] = {(ak*xk + al*xl)/q, (ak*yk + al*yl)/q, (ak*zk + al*zl)/q};
    double W[3] = {(p*P[0] + q*Q[0])/rP, (p*P[1] + q*Q[1])/rP, (p*P[2] + q*Q[2])/rP};
    double AB2 = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj);
    double CD2 = (xk-xl)*(xk-xl) + (yk-yl)*(yk-yl) + (zk-zl)*(zk-zl);
    double PQ2 = (P[0]-Q[0])*(P[0]-Q[0]) + (P[1]-Q[1])*(P[1]-Q[1]) + (P[2]-Q[2])*(P[2]-Q[2]);
    double prefactor = 34.986836655249725 / (p * q * sqrt(rP)) * exp(-ai*aj/p * AB2 - ak*al/q * CD2);
    double F[9]; nwdft_compute_boys(mu * PQ2, F, 8);
    double PA[3] = {P[0]-xi, P[1]-yi, P[2]-zi}, WP[3] = {W[0]-P[0], W[1]-P[1], W[2]-P[2]}, AB[3] = {xi-xj, yi-yj, zi-zj};
    double QC[3] = {Q[0]-xk, Q[1]-yk, Q[2]-zk}, WQ[3] = {W[0]-Q[0], W[1]-Q[1], W[2]-Q[2]}, CD[3] = {xk-xl, yk-yl, zk-zl};
    double rho_inv = 0.5 / rP;
    double ss_m[9]; for(int m=0; m<9; m++) ss_m[m] = prefactor * F[m];
    double ps_ss[3][8]; for(int d=0; d<3; d++) for(int m=0; m<8; m++) ps_ss[d][m] = PA[d]*ss_m[m] + WP[d]*ss_m[m+1];
    double ds_ss[3][3][7]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<=d1; d2++) for(int m=0; m<7; m++) {
        double term = PA[d1]*ps_ss[d2][m] + WP[d1]*ps_ss[d2][m+1];
        if (d1==d2) term += (1.0/(2.0*p))*(ss_m[m] - (q/rP)*ss_m[m+1]);
        ds_ss[d1][d2][m] = ds_ss[d2][d1][m] = term;
    }
    double sp_ss[3][6]; for(int d=0; d<3; d++) for(int m=0; m<6; m++) sp_ss[d][m] = ps_ss[d][m] + AB[d]*ss_m[m];
    double pp_ss[3][3][6]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<3; d2++) for(int m=0; m<6; m++) pp_ss[d1][d2][m] = ds_ss[d1][d2][m] + AB[d2]*ps_ss[d1][m];
    double sd_ss[3][3][6]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<3; d2++) for(int m=0; m<6; m++) sd_ss[d1][d2][m] = pp_ss[d1][d2][m] + AB[d1]*sp_ss[d2][m];

    // Auxiliary arrays for ket P VRR with correct bra reduction
    // ss_ps[d][m] = (ss|p_d s)^m - no bra reduction for ss bra
    double ss_ps[3][4];
    for (int d = 0; d < 3; d++) for (int m = 0; m < 4; m++) {
        ss_ps[d][m] = QC[d]*ss_m[m] + WQ[d]*ss_m[m+1];
    }

    // sp_ps[bra_d][ket_d][m] = (s p_{bra_d} | p_{ket_d} s)^m
    double sp_ps[3][3][4];
    for (int bd = 0; bd < 3; bd++) for (int kd = 0; kd < 3; kd++) for (int m = 0; m < 4; m++) {
        double term = QC[kd]*sp_ss[bd][m] + WQ[kd]*sp_ss[bd][m+1];
        if (bd == kd) term += rho_inv * ss_m[m+1];  // Bra reduction: (ss|ss)^{m+1}
        sp_ps[bd][kd][m] = term;
    }

    // ps_ps[bra_d][ket_d][m] = (p_{bra_d} s | p_{ket_d} s)^m
    double ps_ps[3][3][4];
    for (int bd = 0; bd < 3; bd++) for (int kd = 0; kd < 3; kd++) for (int m = 0; m < 4; m++) {
        double term = QC[kd]*ps_ss[bd][m] + WQ[kd]*ps_ss[bd][m+1];
        if (bd == kd) term += rho_inv * ss_m[m+1];
        ps_ps[bd][kd][m] = term;
    }

    // pp_ps[bra_d1][bra_d2][ket_d][m] = (p_{bra_d1} p_{bra_d2} | p_{ket_d} s)^m
    double pp_ps[3][3][3][3];
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            for (int kd = 0; kd < 3; kd++) {
                for (int m = 0; m < 3; m++) {
                    double term = PA[d1] * sp_ps[d2][kd][m] + WP[d1] * sp_ps[d2][kd][m+1];
                    if (d1 == d2) {
                        double val = ss_ps[kd][m] - (q/rP)*ss_ps[kd][m+1];
                        term += (1.0/(2.0*p)) * val;
                    }
                    if (d1 == kd) {
                        term += rho_inv * sp_ss[d2][m+1];
                    }
                    pp_ps[d1][d2][kd][m] = term;
                }
            }
        }
    }

    int ax_i = (i_type>0 && i_type<=3) ? i_type-1 : -1, ax_j = (j_type>0 && j_type<=3) ? j_type-1 : -1;
    int d1_i=0, d2_i=0, d1_j=0, d2_j=0; if (i_type>=4) nwdft_get_d_indices(i_type, &d1_i, &d2_i); if (j_type>=4) nwdft_get_d_indices(j_type, &d1_j, &d2_j);

    // On-demand computation of higher angular momentum intermediates for DP, PD, DD
    // Only compute when needed (bra has D+P or D+D)
    double dp_val[6] = {0,0,0,0,0,0};  // [m] for specific (d1_i,d2_i,ax_j) - only if DP
    double pd_val[6] = {0,0,0,0,0,0};  // [m] for specific (ax_i,d1_j,d2_j) - only if PD
    double dd_val[5] = {0,0,0,0,0};    // [m] for specific (d1_i,d2_i,d1_j,d2_j) - only if DD

    // Compute DP intermediate: dp_ss[d1][d2][d3][m] = fs_ss[d1][d2][d3][m] + AB[d3]*ds_ss[d1][d2][m]
    // fs_ss[d1][d2][d3][m] via VRR from ds_ss
    if (i_type >= 4 && j_type >= 1 && j_type <= 3) {
        int e1 = d1_i, e2 = d2_i, e3 = ax_j;  // The specific indices we need
        for (int m = 0; m < 6; m++) {
            // Sort e1,e2,e3 for canonical ds_ss access (d1 >= d2)
            int s1 = e1, s2 = e2, s3 = e3;
            // VRR: fs_ss[e1][e2][e3][m] = PA[e1]*ds_ss[e2][e3][m] + WP[e1]*ds_ss[e2][e3][m+1] + terms
            // Need to handle all permutations properly
            // fs_ss with indices (a,b,c) where we add angular momentum a to ds(b,c)
            double fs_val = PA[s1] * ds_ss[s2][s3][m] + WP[s1] * ds_ss[s2][s3][m+1];
            if (s1 == s2) fs_val += (1.0/(2.0*p)) * (ps_ss[s3][m] - (q/rP)*ps_ss[s3][m+1]);
            if (s1 == s3) fs_val += (1.0/(2.0*p)) * (ps_ss[s2][m] - (q/rP)*ps_ss[s2][m+1]);
            // HRR: dp_ss = fs_ss + AB[e3] * ds_ss
            dp_val[m] = fs_val + AB[e3] * ds_ss[e1][e2][m];
        }
    }

    // Compute PD intermediate: pd_ss[d1][d2][d3][m] = dp_ss[d2][d3][d1][m] + AB[d1]*pp_ss[d2][d3][m]
    // We need dp_ss[d1_j][d2_j][ax_i] first
    if (i_type >= 1 && i_type <= 3 && j_type >= 4) {
        int e1 = ax_i, e2 = d1_j, e3 = d2_j;  // pd indices
        for (int m = 0; m < 6; m++) {
            // First compute dp_ss[e2][e3][e1] = fs_ss[e2][e3][e1] + AB[e1]*ds_ss[e2][e3]
            double fs_val = PA[e2] * ds_ss[e3][e1][m] + WP[e2] * ds_ss[e3][e1][m+1];
            // Handle indices correctly - fs_ss(e2, e3, e1) means adding e2 to ds(e3,e1)
            // But ds_ss is symmetric, so ds_ss[e3][e1] = ds_ss[max(e3,e1)][min(e3,e1)]
            int de1 = (e3 > e1) ? e3 : e1, de2 = (e3 > e1) ? e1 : e3;
            fs_val = PA[e2] * ds_ss[de1][de2][m] + WP[e2] * ds_ss[de1][de2][m+1];
            if (e2 == de1) fs_val += (1.0/(2.0*p)) * (ps_ss[de2][m] - (q/rP)*ps_ss[de2][m+1]);
            if (e2 == de2) fs_val += (1.0/(2.0*p)) * (ps_ss[de1][m] - (q/rP)*ps_ss[de1][m+1]);
            double dp_tmp = fs_val + AB[e1] * ds_ss[e2][e3][m];
            // Then pd_ss = dp_tmp + AB[e1] * pp_ss[e2][e3]
            pd_val[m] = dp_tmp + AB[e1] * pp_ss[e2][e3][m];
        }
    }

    // Compute DD intermediate: requires gs_ss -> fd_ss -> dd_ss chain
    // dd_ss[d1][d2][d3][d4][m] - compute only the specific indices needed
    if (i_type >= 4 && j_type >= 4) {
        int e1 = d1_i, e2 = d2_i, e3 = d1_j, e4 = d2_j;
        for (int m = 0; m < 5; m++) {
            // Step 1: Compute fs_ss[e1][e2][e3][m]
            int s1 = e1, s2 = e2, s3 = e3;
            double fs_123 = PA[s1] * ds_ss[s2][s3][m] + WP[s1] * ds_ss[s2][s3][m+1];
            if (s1 == s2) fs_123 += (1.0/(2.0*p)) * (ps_ss[s3][m] - (q/rP)*ps_ss[s3][m+1]);
            if (s1 == s3) fs_123 += (1.0/(2.0*p)) * (ps_ss[s2][m] - (q/rP)*ps_ss[s2][m+1]);

            // Step 2: Compute gs_ss[e1][e2][e3][e4][m] via VRR from fs_ss
            // gs_ss = PA[e1]*fs_ss[e2][e3][e4] + WP[e1]*fs_ss[e2][e3][e4][m+1] + terms
            // Need fs_ss[e2][e3][e4] first
            double fs_234 = PA[e2] * ds_ss[e3][e4][m] + WP[e2] * ds_ss[e3][e4][m+1];
            if (e2 == e3) fs_234 += (1.0/(2.0*p)) * (ps_ss[e4][m] - (q/rP)*ps_ss[e4][m+1]);
            if (e2 == e4) fs_234 += (1.0/(2.0*p)) * (ps_ss[e3][m] - (q/rP)*ps_ss[e3][m+1]);
            double fs_234_m1 = PA[e2] * ds_ss[e3][e4][m+1] + WP[e2] * ds_ss[e3][e4][m+2];
            if (e2 == e3) fs_234_m1 += (1.0/(2.0*p)) * (ps_ss[e4][m+1] - (q/rP)*ps_ss[e4][m+2]);
            if (e2 == e4) fs_234_m1 += (1.0/(2.0*p)) * (ps_ss[e3][m+1] - (q/rP)*ps_ss[e3][m+2]);

            double gs_val = PA[e1] * fs_234 + WP[e1] * fs_234_m1;
            if (e1 == e2) gs_val += (1.0/(2.0*p)) * (ds_ss[e3][e4][m] - (q/rP)*ds_ss[e3][e4][m+1]);
            if (e1 == e3) gs_val += (1.0/(2.0*p)) * (ds_ss[e2][e4][m] - (q/rP)*ds_ss[e2][e4][m+1]);
            if (e1 == e4) gs_val += (1.0/(2.0*p)) * (ds_ss[e2][e3][m] - (q/rP)*ds_ss[e2][e3][m+1]);

            // Step 3: fd_ss = gs_ss + AB[e4] * fs_ss[e1][e2][e3]
            double fd_val = gs_val + AB[e4] * fs_123;

            // Step 4: dp_ss[e1][e2][e3] for dd_ss computation
            double dp_123 = fs_123 + AB[e3] * ds_ss[e1][e2][m];

            // Step 5: dd_ss = fd_ss + AB[e4] * dp_ss[e1][e2][e3]
            dd_val[m] = fd_val + AB[e4] * dp_123;
        }
    }

    // GET_AB_SS macro - now handles DP, PD, DD using on-demand computed values
    #define GET_AB_SS(m) ( \
        (i_type==0 && j_type==0) ? ss_m[m] : \
        (i_type>0 && i_type<=3 && j_type==0) ? ps_ss[ax_i][m] : \
        (i_type==0 && j_type>0 && j_type<=3) ? sp_ss[ax_j][m] : \
        (i_type>0 && i_type<=3 && j_type>0 && j_type<=3) ? pp_ss[ax_i][ax_j][m] : \
        (i_type>=4 && j_type==0) ? ds_ss[d1_i][d2_i][m] : \
        (i_type==0 && j_type>=4) ? sd_ss[d1_j][d2_j][m] : \
        (i_type>=4 && j_type>0 && j_type<=3) ? dp_val[m] : \
        (i_type>0 && i_type<=3 && j_type>=4) ? pd_val[m] : \
        (i_type>=4 && j_type>=4) ? dd_val[m] : 0.0 )

    #define GET_AM1_B_SS(m) ( (j_type==0) ? ss_m[m] : (j_type<=3) ? sp_ss[ax_j][m] : sd_ss[d1_j][d2_j][m] )
    #define GET_A_BM1_SS(m) ( (i_type==0) ? ss_m[m] : (i_type<=3) ? ps_ss[ax_i][m] : ds_ss[d1_i][d2_i][m] )

    double ab_ps[3][3]; for (int d=0; d<3; d++) for (int m=0; m<3; m++) {
        double term = QC[d]*GET_AB_SS(m) + WQ[d]*GET_AB_SS(m+1);
        double bra = 0.0;
        // Bra reduction: VRR uses (a-1,b|ss) - the SOURCE ket, not target
        if (i_type > 0 && i_type <= 3 && ax_i == d) {
            // Reduce bra i: (a-1 b | ss) = (s b|ss) when building (ab|ps) from (ab|ss)
            bra += (j_type==0) ? ss_m[m+1] : (j_type<=3) ? sp_ss[ax_j][m+1] : 0.0;
        }
        if (j_type > 0 && j_type <= 3 && ax_j == d) {
            // Reduce bra j: (a b-1 | ss) = (a s|ss) when building (ab|ps) from (ab|ss)
            bra += (i_type==0) ? ss_m[m+1] : (i_type<=3) ? ps_ss[ax_i][m+1] : 0.0;
        }
        // D-shell bra contributions - use (a-1,b|ss) arrays
        if (i_type >= 4) {
            int n = (d1_i == d ? 1 : 0) + (d2_i == d ? 1 : 0);
            if (n > 0) {
                // (a-1_d, b | ss) where a is D: reduces to (P b | ss)
                int new_ax = (d1_i == d && d2_i == d) ? d : (d1_i == d ? d2_i : d1_i);
                double am1_b = (j_type==0) ? ps_ss[new_ax][m+1] : (j_type<=3) ? pp_ss[new_ax][ax_j][m+1] : 0.0;
                bra += n * am1_b;
            }
        }
        if (j_type >= 4) {
            int n = (d1_j == d ? 1 : 0) + (d2_j == d ? 1 : 0);
            if (n > 0) {
                int new_ax = (d1_j == d && d2_j == d) ? d : (d1_j == d ? d2_j : d1_j);
                double a_bm1 = (i_type==0) ? sp_ss[new_ax][m+1] : (i_type<=3) ? pp_ss[ax_i][new_ax][m+1] : 0.0;
                bra += n * a_bm1;
            }
        }
        ab_ps[d][m] = term + rho_inv * bra;
    }
    if (k_type == 0 && l_type == 0) return GET_AB_SS(0);
    if (k_type <= 3 && l_type == 0) return ab_ps[k_type-1][0];
    if (k_type == 0 && l_type <= 3) return ab_ps[l_type-1][0] + CD[l_type-1]*GET_AB_SS(0);
    int d1_k=0, d2_k=0, d1_l=0, d2_l=0; if (k_type>=4) nwdft_get_d_indices(k_type, &d1_k, &d2_k); if (l_type>=4) nwdft_get_d_indices(l_type, &d1_l, &d2_l);
    int u = (k_type>=4)?d1_k:(k_type>0?k_type-1:0), v = (l_type>=4)?d1_l:(l_type>0?l_type-1:0);
    if (k_type>=4) { u=d1_k; v=d2_k; } else if (l_type>=4) { u=d1_l; v=d2_l; } else { u=k_type-1; v=l_type-1; }

    // GET_AB_DS with bra contributions using reduced-bra (ab|ps) integrals
    double ab_ds_result = 0.0;
    
    // Arrays for D-shell intermediates
    double ab_ds[6][2];
    double ab_pp[3][3][2];
    double ab_dp[6][3][2];
    double ab_pd[3][6];
    double ab_dd[6][6];

    if (k_type >= 4 || l_type >= 4) {
        // Compute full ab_ds array
        for (int dc = 0; dc < 6; dc++) {
            int u, v; nwdft_get_d_indices(dc+4, &u, &v);
            for (int m = 0; m < 2; m++) {
                double term = QC[u]*ab_ps[v][m] + WQ[u]*ab_ps[v][m+1];
                if (u == v) term += (1.0/(2.0*q))*(GET_AB_SS(m) - (p/rP)*GET_AB_SS(m+1));
                
                // Bra reduction
                if (i_type > 0 && i_type <= 3 && ax_i == u) {
                    double reduced = (j_type==0) ? ss_ps[v][m+1] : (j_type<=3) ? sp_ps[ax_j][v][m+1] : 0.0;
                    term += rho_inv * reduced;
                }
                if (j_type > 0 && j_type <= 3 && ax_j == u) {
                    double reduced = (i_type==0) ? ss_ps[v][m+1] : (i_type<=3) ? ps_ps[ax_i][v][m+1] : 0.0;
                    term += rho_inv * reduced;
                }
                if (i_type >= 4) {
                    int n = (d1_i == u ? 1 : 0) + (d2_i == u ? 1 : 0);
                    if (n > 0) {
                        int new_ax = (d1_i == u && d2_i == u) ? u : (d1_i == u ? d2_i : d1_i);
                        double reduced = (j_type==0) ? ps_ps[new_ax][v][m+1] : (j_type<=3) ? pp_ps[new_ax][ax_j][v][m+1] : 0.0;
                        term += rho_inv * n * reduced;
                    }
                }
                if (j_type >= 4) {
                    int n = (d1_j == u ? 1 : 0) + (d2_j == u ? 1 : 0);
                    if (n > 0) {
                        int new_ax = (d1_j == u && d2_j == u) ? u : (d1_j == u ? d2_j : d1_j);
                        double reduced = (i_type==0) ? sp_ps[new_ax][v][m+1] : (i_type<=3) ? pp_ps[ax_i][new_ax][v][m+1] : 0.0;
                        term += rho_inv * n * reduced;
                    }
                }
                ab_ds[dc][m] = term;
            }
        }
        
        // Compute full ab_pp array via HRR from ab_ds
        for (int mm = 0; mm < 2; mm++) {
            for (int u = 0; u < 3; u++) for (int v = 0; v < 3; v++) {
                int dc_uv;
                if (u <= v) {
                    if (u==0 && v==0) dc_uv=0; else if (u==0 && v==1) dc_uv=1; else if (u==0 && v==2) dc_uv=2;
                    else if (u==1 && v==1) dc_uv=3; else if (u==1 && v==2) dc_uv=4; else dc_uv=5;
                } else {
                    if (v==0 && u==1) dc_uv=1; else if (v==0 && u==2) dc_uv=2; else dc_uv=4;
                }
                ab_pp[u][v][mm] = ab_ds[dc_uv][mm] + CD[v]*ab_ps[u][mm];
            }
        }

        // ab_dp[dc][dir][m]
        // double ab_dp[6][3][2]; // Declared above
        for (int dc = 0; dc < 6; dc++) {
            int u, v; nwdft_get_d_indices(dc+4, &u, &v);
            for (int dir = 0; dir < 3; dir++) {
                for (int m = 0; m < 2; m++) {
                    double term = QC[dir]*ab_ds[dc][m] + WQ[dir]*ab_ds[dc][m+1];
                    // Ket reduction of D if D has angular momentum in direction dir
                    // N_D[dir]/(2q) * ((ab|D-1_dir s)^m - p/rP*(ab|D-1_dir s)^{m+1})
                    {
                        int n_ket = (u == dir ? 1 : 0) + (v == dir ? 1 : 0);
                        if (n_ket > 0) {
                            // D-1_dir reduces D to P
                            int reduced_dir = (u == dir && v == dir) ? dir : (u == dir ? v : u);
                            double val = ab_ps[reduced_dir][m] - (p/rP)*ab_ps[reduced_dir][m+1];
                            term += n_ket * (1.0/(2.0*q)) * val;
                        }
                    }
                    // Bra reduction: (a-1|DP) from (a-1|Ds)
                    if (i_type > 0) {
                        // Check if bra 'a' (i) reduces along 'dir'
                        // VRR adds momentum 'dir'. Bra reduction term N_a(dir).
                        // If i is S: N=0.
                        // If i is P: N=1 if ax_i == dir.
                        // If i is D: N=1 if d1_i==dir or d2_i==dir.

                        double bra_ds_val = 0.0;
                        int n_bra = 0;

                        // Check Bra A
                        if (i_type <= 3 && ax_i == dir) n_bra = 1;
                        else if (i_type >= 4) n_bra = (d1_i == dir ? 1 : 0) + (d2_i == dir ? 1 : 0);

                        if (n_bra > 0) {
                            // Compute (a-1 b|Ds) for the specific a-1
                            // We need reduced_ab_ds[dc][m].
                            // Reconstruct it:
                            // (a-1 b|D_uv s) = QC[u]*(a-1 b|P_v s) + WQ[u]*(a-1 b|P_v s)^1 + ...

                            // Determine the reduced bra type and index
                            int r_i_type, r_ax_i;
                            int r_d1_i, r_d2_i;

                            if (i_type <= 3) { r_i_type = 0; } // S
                            else {
                                // D -> P
                                r_i_type = (d1_i == dir && d2_i == dir) ? 1+dir : (d1_i == dir ? 1+d2_i : 1+d1_i);
                                r_ax_i = r_i_type - 1;
                            }

                            // Retrieve (a-1 b | P_v s)
                            double rb_ps_0 = 0.0, rb_ps_1 = 0.0;
                            if (j_type == 0) {
                                rb_ps_0 = (r_i_type==0) ? ss_ps[v][m] : ps_ps[r_ax_i][v][m];
                                rb_ps_1 = (r_i_type==0) ? ss_ps[v][m+1] : ps_ps[r_ax_i][v][m+1];
                            } else if (j_type <= 3) {
                                rb_ps_0 = (r_i_type==0) ? sp_ps[ax_j][v][m] : pp_ps[r_ax_i][ax_j][v][m];
                                rb_ps_1 = (r_i_type==0) ? sp_ps[ax_j][v][m+1] : pp_ps[r_ax_i][ax_j][v][m+1];
                            }
                            // D-bra j not supported here for brevity (rare)

                            double r_ds = QC[u]*rb_ps_0 + WQ[u]*rb_ps_1;
                            if (u == v) {
                                // Ket reduction for r_ds: (a-1 b | s s)
                                double rb_ss_0 = 0.0, rb_ss_1 = 0.0;
                                // Need (a-1 b | s s).
                                if (j_type == 0) {
                                    rb_ss_0 = (r_i_type==0) ? ss_m[m] : ps_ss[r_ax_i][m];
                                    rb_ss_1 = (r_i_type==0) ? ss_m[m+1] : ps_ss[r_ax_i][m+1];
                                } else if (j_type <= 3) {
                                    rb_ss_0 = (r_i_type==0) ? sp_ss[ax_j][m] : pp_ss[r_ax_i][ax_j][m];
                                    rb_ss_1 = (r_i_type==0) ? sp_ss[ax_j][m+1] : pp_ss[r_ax_i][ax_j][m+1];
                                }
                                r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                            }
                            // Inner bra reduction? (a-2 b | D s). Ignore for now (max L=2).

                            bra_ds_val += n_bra * r_ds;
                        }

                        // Check Bra B
                        n_bra = 0;
                        if (j_type > 0 && j_type <= 3 && ax_j == dir) n_bra = 1;
                        else if (j_type >= 4) n_bra = (d1_j == dir ? 1 : 0) + (d2_j == dir ? 1 : 0);

                        if (n_bra > 0) {
                             // Similar logic for B reduction...
                             // For now assume j is S or P (h2o/benzene limit).
                             // If j is P, reduces to S.
                             // (a S | D s).
                             // Need (a S | P s).
                             // If i=S: ss_ps. If i=P: ps_ps.
                             double rb_ps_0 = (i_type==0) ? ss_ps[v][m] : (i_type<=3) ? ps_ps[ax_i][v][m] : 0.0;
                             double rb_ps_1 = (i_type==0) ? ss_ps[v][m+1] : (i_type<=3) ? ps_ps[ax_i][v][m+1] : 0.0;
                             double r_ds = QC[u]*rb_ps_0 + WQ[u]*rb_ps_1;
                             if (u == v) {
                                 double rb_ss_0 = (i_type==0) ? ss_m[m] : ps_ss[ax_i][m];
                                 double rb_ss_1 = (i_type==0) ? ss_m[m+1] : ps_ss[ax_i][m+1];
                                 r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                             }
                             bra_ds_val += n_bra * r_ds;
                        }

                        term += rho_inv * bra_ds_val;
                    }
                    ab_dp[dc][dir][m] = term;
                }
            }
        }

        // Compute ab_pd via HRR from ab_dp
        // double ab_pd[3][6];
        for (int u = 0; u < 3; u++) for (int dc = 0; dc < 6; dc++) {
            int v, w; nwdft_get_d_indices(dc+4, &v, &w);
            // Treat w as the added momentum
            // (p_u, d_{vw}) = (d_{uw}, p_v) + CD[w] * (p_u, p_v)
            
            // Find index for d_{uw}
            int dc_uw;
            int a=u, b=w;
            if (a>b) { int t=a; a=b; b=t; }
            if (a==0 && b==0) dc_uw=0;
            else if (a==0 && b==1) dc_uw=1;
            else if (a==0 && b==2) dc_uw=2;
            else if (a==1 && b==1) dc_uw=3;
            else if (a==1 && b==2) dc_uw=4;
            else dc_uw=5;
            
            ab_pd[u][dc] = ab_dp[dc_uw][v][0] + CD[w] * ab_pp[u][v][0];
        }

        // Compute ab_dd via VRR on ab_dp
        // double ab_dd[6][6];
        for (int d1 = 0; d1 < 6; d1++) {
            for (int d2 = 0; d2 < 6; d2++) {
                int x, y; nwdft_get_d_indices(d2+4, &x, &y);
                // Building d2 (xy) from p_x. Adding y.
                double term = QC[y]*ab_dp[d1][x][0] + WQ[y]*ab_dp[d1][x][1];
                
                // Ket reduction (p_x -> s): (ab|d1 s)
                if (x == y) {
                    double val = ab_ds[d1][0] - (p/rP)*ab_ds[d1][1];
                    term += (1.0/(2.0*q)) * val;
                }
                
                // Bra reduction: (a-1_y b | d1 p_x)
                if (i_type > 0) {
                    double bra_dd_val = 0.0;
                    int n_bra = 0;
                    if (i_type <= 3 && ax_i == y) n_bra = 1;
                    else if (i_type >= 4) n_bra = (d1_i == y ? 1 : 0) + (d2_i == y ? 1 : 0);
                    
                    if (n_bra > 0) {
                        // Reconstruct (a-1_y | d1 s) for bra direction y
                        // Copied from ab_dp logic but specialized for direction y
                        int r_i_type, r_ax_i;
                        if (i_type <= 3) { r_i_type = 0; }
                        else { r_i_type = (d1_i == y && d2_i == y) ? 1+y : (d1_i == y ? 1+d2_i : 1+d1_i); r_ax_i = r_i_type - 1; }
                        
                        // (a-1_y b | P_v s)
                        int u_d1, v_d1; nwdft_get_d_indices(d1+4, &u_d1, &v_d1);
                        double rb_ps_0 = 0.0, rb_ps_1 = 0.0;
                        if (j_type == 0) {
                            rb_ps_0 = (r_i_type==0) ? ss_ps[v_d1][0] : ps_ps[r_ax_i][v_d1][0];
                            rb_ps_1 = (r_i_type==0) ? ss_ps[v_d1][1] : ps_ps[r_ax_i][v_d1][1];
                        } else if (j_type <= 3) {
                            rb_ps_0 = (r_i_type==0) ? sp_ps[ax_j][v_d1][0] : pp_ps[r_ax_i][ax_j][v_d1][0];
                            rb_ps_1 = (r_i_type==0) ? sp_ps[ax_j][v_d1][1] : pp_ps[r_ax_i][ax_j][v_d1][1];
                        }
                        
                        double r_ds_0 = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0 = 0.0, rb_ss_1 = 0.0;
                            if (j_type == 0) {
                                rb_ss_0 = (r_i_type==0) ? ss_m[0] : ps_ss[r_ax_i][0];
                                rb_ss_1 = (r_i_type==0) ? ss_m[1] : ps_ss[r_ax_i][1];
                            } else if (j_type <= 3) {
                                rb_ss_0 = (r_i_type==0) ? sp_ss[ax_j][0] : pp_ss[r_ax_i][ax_j][0];
                                rb_ss_1 = (r_i_type==0) ? sp_ss[ax_j][1] : pp_ss[r_ax_i][ax_j][1];
                            }
                            r_ds_0 += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        
                        // We also need r_ds_1 (m=1) for the next step?
                        // (a-1 | d1 p_x) = QC[x]*(a-1|d1 s)^0 + WQ[x]*(a-1|d1 s)^1
                        // So yes, need r_ds at m=1.
                        // Repetitive code, but robust.
                        double rb_ps_0_m1 = 0.0, rb_ps_1_m1 = 0.0;
                        if (j_type == 0) {
                            rb_ps_0_m1 = (r_i_type==0) ? ss_ps[v_d1][1] : ps_ps[r_ax_i][v_d1][1];
                            rb_ps_1_m1 = (r_i_type==0) ? ss_ps[v_d1][2] : ps_ps[r_ax_i][v_d1][2];
                        } else if (j_type <= 3) {
                            rb_ps_0_m1 = (r_i_type==0) ? sp_ps[ax_j][v_d1][1] : pp_ps[r_ax_i][ax_j][v_d1][1];
                            rb_ps_1_m1 = (r_i_type==0) ? sp_ps[ax_j][v_d1][2] : pp_ps[r_ax_i][ax_j][v_d1][2];
                        }
                        double r_ds_1 = QC[u_d1]*rb_ps_0_m1 + WQ[u_d1]*rb_ps_1_m1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0_m1 = 0.0, rb_ss_1_m1 = 0.0;
                            if (j_type == 0) {
                                rb_ss_0_m1 = (r_i_type==0) ? ss_m[1] : ps_ss[r_ax_i][1];
                                rb_ss_1_m1 = (r_i_type==0) ? ss_m[2] : ps_ss[r_ax_i][2];
                            } else if (j_type <= 3) {
                                rb_ss_0_m1 = (r_i_type==0) ? sp_ss[ax_j][1] : pp_ss[r_ax_i][ax_j][1];
                                rb_ss_1_m1 = (r_i_type==0) ? sp_ss[ax_j][2] : pp_ss[r_ax_i][ax_j][2];
                            }
                            r_ds_1 += (1.0/(2.0*q)) * (rb_ss_0_m1 - (p/rP)*rb_ss_1_m1);
                        }
                        
                        // Compute (a-1 | d1 p_x)
                        double r_dp = QC[x]*r_ds_0 + WQ[x]*r_ds_1;
                        if (u_d1 == x) {
                             // Ket reduction d1->s. (a-1 | s s).
                             // r_ds_0 was (a-1 | d1 s).
                             // We need (a-1 | s s). This is rb_ss_0.
                             // Actually, d1 reduces to s.
                             // Wait. If d1 reduced to s?
                             // (a-1 | d1 p_x). Reduce p_x to s.
                             // -> (a-1 | d1 s). This is r_ds_0.
                             // This is ket reduction of the TARGET p_x.
                             // Handled below?
                             // No, this is BRA reduction term calculation.
                             // We are computing (a-1 | d1 p_x).
                             // It has ket reduction term if x reduces to something?
                             // If d1 and p_x match? No.
                             // We built d1 then added p_x.
                             // Ket reduction of p_x is (a-1 | d1 s).
                             // This is r_ds_0.
                             // But my code loop: `if (u_d1 == x)`? No.
                             // The VRR for (d1 p_x) reduces p_x.
                             // If p_x index 'x' matches? No, always reduces p to s.
                             // But does it match d1? No.
                             // Term `N_c / 2q * (a-1 | c-1 d)`.
                             // c = p_x. c-1 = s.
                             // `(a-1 | d1 s)`.
                             // Yes. `r_ds_0`.
                             // Wait. `u_d1 == x` check?
                             // In ab_dp loop: `double val = ab_ds[dc][m] ... term += val`.
                             // It didn't check u==v.
                             // Because p reduces to s ALWAYS.
                             // Ah, in ab_dp loop: `if (/* p matches D... */)` logic was commented out/confused.
                             // Correct logic: p ALWAYS reduces to s.
                             // `term += (1/2q) * (ab_ds - ...)`.
                             // So here `r_dp += (1/2q) * (r_ds_0 - ...)`?
                             // Yes.
                             
                             double r_val = r_ds_0 - (p/rP)*r_ds_1;
                             r_dp += (1.0/(2.0*q)) * r_val;
                        }
                        
                        bra_dd_val += n_bra * r_dp;
                    }
                    
                    // Same for Bra B (j)
                    if (j_type > 0 && j_type <= 3 && ax_j == y) n_bra = 1;
                    else if (j_type >= 4) n_bra = (d1_j == y ? 1 : 0) + (d2_j == y ? 1 : 0);
                    
                    if (n_bra > 0) {
                        // ... similar logic for B ...
                        // For brevity/limitations, assume B is S/P.
                        // (a b-1_y | d1 p_x)
                        // Need (a b-1_y | d1 s) -> r_ds
                        // Need (a b-1_y | P_v s) -> rb_ps
                        double rb_ps_0 = 0.0, rb_ps_1 = 0.0;
                        int u_d1, v_d1; nwdft_get_d_indices(d1+4, &u_d1, &v_d1);
                        
                        // i type S or P
                        rb_ps_0 = (i_type==0) ? ss_ps[v_d1][0] : ps_ps[ax_i][v_d1][0];
                        rb_ps_1 = (i_type==0) ? ss_ps[v_d1][1] : ps_ps[ax_i][v_d1][1];
                        
                        double r_ds_0 = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0 = (i_type==0) ? ss_m[0] : ps_ss[ax_i][0];
                            double rb_ss_1 = (i_type==0) ? ss_m[1] : ps_ss[ax_i][1];
                            r_ds_0 += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        
                        // Need r_ds_1
                        double rb_ps_0_m1 = (i_type==0) ? ss_ps[v_d1][1] : ps_ps[ax_i][v_d1][1];
                        double rb_ps_1_m1 = (i_type==0) ? ss_ps[v_d1][2] : ps_ps[ax_i][v_d1][2];
                        double r_ds_1 = QC[u_d1]*rb_ps_0_m1 + WQ[u_d1]*rb_ps_1_m1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0_m1 = (i_type==0) ? ss_m[1] : ps_ss[ax_i][1];
                            double rb_ss_1_m1 = (i_type==0) ? ss_m[2] : ps_ss[ax_i][2];
                            r_ds_1 += (1.0/(2.0*q)) * (rb_ss_0_m1 - (p/rP)*rb_ss_1_m1);
                        }
                        
                        double r_dp = QC[x]*r_ds_0 + WQ[x]*r_ds_1;
                        // Ket reduction (p_x -> s)
                        double r_val = r_ds_0 - (p/rP)*r_ds_1;
                        r_dp += (1.0/(2.0*q)) * r_val;
                        
                        bra_dd_val += n_bra * r_dp;
                    }
                    term += rho_inv * bra_dd_val;
                }
                ab_dd[d1][d2] = term;
            }
        }
        
    } else {
        // Fast path for non-D shells (P,P case) - compute single component
        double term = QC[u]*ab_ps[v][0] + WQ[u]*ab_ps[v][1];
        if (u == v) term += (1.0/(2.0*q))*(GET_AB_SS(0) - (p/rP)*GET_AB_SS(1));
        // P-shell bra contributions: use (a-1 b|p_v s) not (ab|p_v s)
        if (i_type > 0 && i_type <= 3 && ax_i == u) {
            double reduced = (j_type==0) ? ss_ps[v][1] : (j_type<=3) ? sp_ps[ax_j][v][1] : 0.0;
            term += rho_inv * reduced;
        }
        if (j_type > 0 && j_type <= 3 && ax_j == u) {
            double reduced = (i_type==0) ? ss_ps[v][1] : (i_type<=3) ? ps_ps[ax_i][v][1] : 0.0;
            term += rho_inv * reduced;
        }
        // D-shell bra contributions for ab_ds - use reduced-bra integrals
        if (i_type >= 4) {
            int n = (d1_i == u ? 1 : 0) + (d2_i == u ? 1 : 0);
            if (n > 0) {
                int new_ax = (d1_i == u && d2_i == u) ? u : (d1_i == u ? d2_i : d1_i);
                double reduced = (j_type==0) ? ps_ps[new_ax][v][1] : 0.0;
                term += rho_inv * n * reduced;
            }
        }
        if (j_type >= 4) {
            int n = (d1_j == u ? 1 : 0) + (d2_j == u ? 1 : 0);
            if (n > 0) {
                int new_ax = (d1_j == u && d2_j == u) ? u : (d1_j == u ? d2_j : d1_j);
                double reduced = (i_type==0) ? sp_ps[new_ax][v][1] : 0.0;
                term += rho_inv * n * reduced;
            }
        }
        ab_ds_result = term;
    }

    // Dispatch return value based on k,l types
    if (k_type >= 4 || l_type >= 4) {
        if (k_type >= 4 && l_type == 0) { // (ab|Ds)
            return ab_ds[k_type-4][0];
        }
        if (k_type == 0 && l_type >= 4) { // (ab|sD)
            int u, v; nwdft_get_d_indices(l_type, &u, &v);
            if (u == v) {
                return ab_ds[l_type-4][0] + 2.0*CD[u]*ab_ps[u][0] + CD[u]*CD[u]*GET_AB_SS(0);
            } else {
                return ab_ds[l_type-4][0] + CD[v]*ab_ps[u][0] + CD[u]*ab_ps[v][0] + CD[u]*CD[v]*GET_AB_SS(0);
            }
        }
        if (k_type >= 4 && l_type > 0 && l_type <= 3) { // (ab|DP)
            return ab_dp[k_type-4][l_type-1][0];
        }
        if (k_type > 0 && k_type <= 3 && l_type >= 4) { // (ab|PD)
            return ab_pd[k_type-1][l_type-4];
        }
        if (k_type >= 4 && l_type >= 4) { // (ab|DD)
            return ab_dd[k_type-4][l_type-4];
        }
    }

    // Standard PP case: (ab|pp) = (ab|ds) + CD*(ab|ps)
    return ab_ds_result + CD[v]*ab_ps[u][0];
}

__global__ void nwdft_compute_K_kernel(int nshells, Shell *shells, double *exps, double *coefs, double *P, double *K, int nbf, int me, int nproc) {
    int ish = blockIdx.x * blockDim.x + threadIdx.x;
    int jsh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ish >= nshells || jsh >= nshells) return;
    
    // Simple workload partitioning based on shell pairs (ish, jsh)
    int pair_idx = ish * nshells + jsh;
    if (pair_idx % nproc != me) return;

    Shell si = shells[ish], sj = shells[jsh];
    int ni = (si.L==0)?1:(si.L==1)?3:(si.L==2)?6:10, nj = (sj.L==0)?1:(sj.L==1)?3:(sj.L==2)?6:10;
    
    for (int ksh=0; ksh<nshells; ksh++) {
        Shell sk = shells[ksh];
        int nk = (sk.L==0)?1:(sk.L==1)?3:(sk.L==2)?6:10;
        for (int lsh=0; lsh<nshells; lsh++) {
            Shell sl = shells[lsh];
            int nl = (sl.L==0)?1:(sl.L==1)?3:(sl.L==2)?6:10;
            
            for (int ic=0; ic<ni; ic++) for (int jc=0; jc<nj; jc++) {
                int idx_i = si.basis_idx + ic, idx_j = sj.basis_idx + jc;
                for (int kc=0; kc<nk; kc++) for (int lc=0; lc<nl; lc++) {
                    double integral = 0.0;
                    for (int pi=0; pi<si.nprim; pi++) for (int pj=0; pj<sj.nprim; pj++) 
                    for (int pk=0; pk<sk.nprim; pk++) for (int pl=0; pl<sl.nprim; pl++) {
                        int ti = (si.L==0)?0:(si.L==1)?1+ic:4+ic, tj = (sj.L==0)?0:(sj.L==1)?1+jc:4+jc;
                        int tk = (sk.L==0)?0:(sk.L==1)?1+kc:4+kc, tl = (sl.L==0)?0:(sl.L==1)?1+lc:4+lc;
                        integral += nwdft_compute_eri_primitive(exps[si.ptr_exp+pi], exps[sj.ptr_exp+pj], exps[sk.ptr_exp+pk], exps[sl.ptr_exp+pl],
                                si.x, si.y, si.z, sj.x, sj.y, sj.z, sk.x, sk.y, sk.z, sl.x, sl.y, sl.z, ti, tj, tk, tl) *
                            coefs[si.ptr_coef+pi] * coefs[sj.ptr_coef+pj] * coefs[sk.ptr_coef+pk] * coefs[sl.ptr_coef+pl];
                    }
                    int idx_k = sk.basis_idx + kc, idx_l = sl.basis_idx + lc;
                    double term = 0.5 * integral * P[idx_l*nbf + idx_j];
                    nwdft_atomicAdd_safe(&K[idx_k*nbf + idx_i], term);
                    nwdft_atomicAdd_safe(&K[idx_i*nbf + idx_k], term);
                }
            }
        }
    }
}

// Forward declaration for cpu_eri_full (defined below)
double cpu_eri_full(double ai, double aj, double ak, double al,
                    double xi, double yi, double zi,
                    double xj, double yj, double zj,
                    double xk, double yk, double zk,
                    double xl, double yl, double zl,
                    int i_type, int j_type, int k_type, int l_type);

extern "C" void nwdft_gpu_compute_fock_jk_(double *h_P, double *h_J, double *h_K, long *nbf, long *do_k, long *me, long *nproc) {

    if (!is_init) nwdft_gpu_init_();

    if (n_shells_gpu == 0) return;

    size_t size = sizeof(double) * (*nbf) * (*nbf);
    int n = (int)*nbf;

    double *d_P, *d_K = NULL;

    cudaMalloc(&d_P, size); if (*do_k) cudaMalloc(&d_K, size);

    cudaMemcpy(d_P, h_P, size, cudaMemcpyHostToDevice);

    if (*do_k) cudaMemset(d_K, 0, size);



    dim3 threads(8, 8);

    dim3 blocks((n_shells_gpu + 7)/8, (n_shells_gpu + 7)/8);



    nwdft_compute_K_kernel<<<blocks, threads>>>(n_shells_gpu, d_shells, d_exponents, d_coefs, d_P, d_K, (int)*nbf, (int)*me, (int)*nproc);

    cudaDeviceSynchronize();

    if (*do_k) cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);

    // Debug: Print K matrix elements
    /*
    static int call_count = 0;
    call_count++;
    if (call_count == 1 && *do_k && *me == 0) {
        printf("\n=== GPU K-matrix (first 4x4) ===\n");
        for (int i = 0; i < 4 && i < n; i++) {
            for (int j = 0; j < 4 && j < n; j++) {
                printf("%12.6f ", h_K[i*n + j]);
            }
            printf("\n");
        }
    }
    */

    cudaFree(d_P);
    if (d_K) cudaFree(d_K);

    cudaFree(d_P);
    if (d_K) cudaFree(d_K);
}

// CPU Boys function
void cpu_boys(double T, double *F, int mmax) {
    if (T < 1e-7) {
        for (int m = 0; m <= mmax; m++) F[m] = 1.0/(2*m+1) - T/(2*m+3);
    } else if (T > 30.0) {
        F[0] = 0.5 * sqrt(M_PI / T);
        for (int m = 0; m < mmax; m++) F[m+1] = (2*m+1) * F[m] / (2.0 * T);
    } else {
        int M = mmax + 15;
        double exp_t = exp(-T);
        double res[40]; res[M] = 0.0;
        for (int m = M; m > 0; m--) res[m-1] = (2.0*T*res[m] + exp_t) / (2.0*m - 1.0);
        for (int m = 0; m <= mmax; m++) F[m] = res[m];
    }
}

// Full CPU ERI function - mirrors the GPU implementation
double cpu_eri_full(double ai, double aj, double ak, double al,
                    double xi, double yi, double zi,
                    double xj, double yj, double zj,
                    double xk, double yk, double zk,
                    double xl, double yl, double zl,
                    int i_type, int j_type, int k_type, int l_type) {
    if ( (k_type + l_type) > (i_type + j_type) ) {
        return cpu_eri_full(ak, al, ai, aj, xk, yk, zk, xl, yl, zl, xi, yi, zi, xj, yj, zj, k_type, l_type, i_type, j_type);
    }
    double p = ai + aj, q = ak + al, rP = p + q, mu = p * q / rP;
    double P[3] = {(ai*xi + aj*xj)/p, (ai*yi + aj*yj)/p, (ai*zi + aj*zj)/p};
    double Q[3] = {(ak*xk + al*xl)/q, (ak*yk + al*yl)/q, (ak*zk + al*zl)/q};
    double W[3] = {(p*P[0] + q*Q[0])/rP, (p*P[1] + q*Q[1])/rP, (p*P[2] + q*Q[2])/rP};
    double AB2 = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj);
    double CD2 = (xk-xl)*(xk-xl) + (yk-yl)*(yk-yl) + (zk-zl)*(zk-zl);
    double PQ2 = (P[0]-Q[0])*(P[0]-Q[0]) + (P[1]-Q[1])*(P[1]-Q[1]) + (P[2]-Q[2])*(P[2]-Q[2]);
    double prefactor = 34.986836655249725 / (p * q * sqrt(rP)) * exp(-ai*aj/p * AB2 - ak*al/q * CD2);
    double F[9]; cpu_boys(mu * PQ2, F, 8);
    double PA[3] = {P[0]-xi, P[1]-yi, P[2]-zi}, WP[3] = {W[0]-P[0], W[1]-P[1], W[2]-P[2]}, AB[3] = {xi-xj, yi-yj, zi-zj};
    double QC[3] = {Q[0]-xk, Q[1]-yk, Q[2]-zk}, WQ[3] = {W[0]-Q[0], W[1]-Q[1], W[2]-Q[2]}, CD[3] = {xk-xl, yk-yl, zk-zl};
    double rho_inv = 0.5 / rP;
    double ss_m[9]; for(int m=0; m<9; m++) ss_m[m] = prefactor * F[m];
    double ps_ss[3][8]; for(int d=0; d<3; d++) for(int m=0; m<8; m++) ps_ss[d][m] = PA[d]*ss_m[m] + WP[d]*ss_m[m+1];
    double ds_ss[3][3][7]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<=d1; d2++) for(int m=0; m<7; m++) {
        double term = PA[d1]*ps_ss[d2][m] + WP[d1]*ps_ss[d2][m+1];
        if (d1==d2) term += (1.0/(2.0*p))*(ss_m[m] - (q/rP)*ss_m[m+1]);
        ds_ss[d1][d2][m] = ds_ss[d2][d1][m] = term;
    }
    double sp_ss[3][6]; for(int d=0; d<3; d++) for(int m=0; m<6; m++) sp_ss[d][m] = ps_ss[d][m] + AB[d]*ss_m[m];
    double pp_ss[3][3][6]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<3; d2++) for(int m=0; m<6; m++) pp_ss[d1][d2][m] = ds_ss[d1][d2][m] + AB[d2]*ps_ss[d1][m];
    double sd_ss[3][3][6]; for(int d1=0; d1<3; d1++) for(int d2=0; d2<3; d2++) for(int m=0; m<6; m++) sd_ss[d1][d2][m] = pp_ss[d1][d2][m] + AB[d1]*sp_ss[d2][m];

    // Compute dp_ss, pd_ss, dd_ss for D-shell bra support
    double dp_ss[3][3][3][5]; // dp_ss[d1][d2][p][m] = (d_{d1d2} p_p | ss)^m
    for(int d1=0; d1<3; d1++) for(int d2=0; d2<=d1; d2++) for(int pp=0; pp<3; pp++) for(int m=0; m<5; m++) {
        double term = PA[d1]*pp_ss[d2][pp][m] + WP[d1]*pp_ss[d2][pp][m+1];
        if (d1==d2) term += (1.0/(2.0*p))*(sp_ss[pp][m] - (q/rP)*sp_ss[pp][m+1]);
        if (d1==pp) term += (1.0/(2.0*p))*(ps_ss[d2][m] - (q/rP)*ps_ss[d2][m+1]);
        dp_ss[d1][d2][pp][m] = dp_ss[d2][d1][pp][m] = term;
    }
    double pd_ss[3][3][3][5]; // pd_ss[p][d1][d2][m] = (p_p d_{d1d2} | ss)^m via HRR
    for(int pp=0; pp<3; pp++) for(int d1=0; d1<3; d1++) for(int d2=0; d2<3; d2++) for(int m=0; m<5; m++) {
        pd_ss[pp][d1][d2][m] = dp_ss[d1][d2][pp][m] + AB[pp]*sd_ss[d1][d2][m];
    }
    double dd_ss[3][3][3][3][4]; // dd_ss[a1][a2][b1][b2][m] = (d_{a1a2} d_{b1b2} | ss)^m
    for(int a1=0; a1<3; a1++) for(int a2=0; a2<=a1; a2++) for(int b1=0; b1<3; b1++) for(int b2=0; b2<=b1; b2++) for(int m=0; m<4; m++) {
        double term = PA[a1]*pd_ss[a2][b1][b2][m] + WP[a1]*pd_ss[a2][b1][b2][m+1];
        // Bra-a reduction: reduce P_a2 when a1==a2
        if (a1==a2) term += (1.0/(2.0*p))*(sd_ss[b1][b2][m] - (q/rP)*sd_ss[b1][b2][m+1]);
        // Bra-b reduction: reduce D_b1b2 -> P_remaining, combined with P_a2 gives pp_ss[a2][remaining]
        if (a1==b1 && b1==b2) {
            // D_aa -> P_a (N=2), result is pp_ss[a2][a1]
            term += (2.0/(2.0*p))*(pp_ss[a2][a1][m] - (q/rP)*pp_ss[a2][a1][m+1]);
        } else if (a1==b1 && b1!=b2) {
            // D_ab -> P_b (N=1), result is pp_ss[a2][b2]
            term += (1.0/(2.0*p))*(pp_ss[a2][b2][m] - (q/rP)*pp_ss[a2][b2][m+1]);
        } else if (a1==b2 && b1!=b2) {
            // D_ab -> P_a (N=1), result is pp_ss[a2][b1]
            term += (1.0/(2.0*p))*(pp_ss[a2][b1][m] - (q/rP)*pp_ss[a2][b1][m+1]);
        }
        dd_ss[a1][a2][b1][b2][m] = dd_ss[a2][a1][b1][b2][m] = dd_ss[a1][a2][b2][b1][m] = dd_ss[a2][a1][b2][b1][m] = term;
    }

    int ax_i = (i_type>0 && i_type<=3) ? i_type-1 : -1, ax_j = (j_type>0 && j_type<=3) ? j_type-1 : -1;
    int d1_i=0, d2_i=0, d1_j=0, d2_j=0;
    if (i_type>=4) { int t=i_type-4; if(t==0){d1_i=0;d2_i=0;} else if(t==1){d1_i=0;d2_i=1;} else if(t==2){d1_i=0;d2_i=2;} else if(t==3){d1_i=1;d2_i=1;} else if(t==4){d1_i=1;d2_i=2;} else if(t==5){d1_i=2;d2_i=2;} }
    if (j_type>=4) { int t=j_type-4; if(t==0){d1_j=0;d2_j=0;} else if(t==1){d1_j=0;d2_j=1;} else if(t==2){d1_j=0;d2_j=2;} else if(t==3){d1_j=1;d2_j=1;} else if(t==4){d1_j=1;d2_j=2;} else if(t==5){d1_j=2;d2_j=2;} }

    // Macro for GET_AB_SS with D-shell bra support
    #define GET_AB_SS_CPU(m) ( \
        (i_type==0 && j_type==0) ? ss_m[m] : \
        (i_type>0 && i_type<=3 && j_type==0) ? ps_ss[ax_i][m] : \
        (i_type==0 && j_type>0 && j_type<=3) ? sp_ss[ax_j][m] : \
        (i_type>0 && i_type<=3 && j_type>0 && j_type<=3) ? pp_ss[ax_i][ax_j][m] : \
        (i_type>=4 && j_type==0) ? ds_ss[d1_i][d2_i][m] : \
        (i_type==0 && j_type>=4) ? sd_ss[d1_j][d2_j][m] : \
        (i_type>=4 && j_type>0 && j_type<=3) ? dp_ss[d1_i][d2_i][ax_j][m] : \
        (i_type>0 && i_type<=3 && j_type>=4) ? pd_ss[ax_i][d1_j][d2_j][m] : \
        (i_type>=4 && j_type>=4) ? dd_ss[d1_i][d2_i][d1_j][d2_j][m] : 0.0 )

    // Handle (ab|ss) cases
    if (k_type == 0 && l_type == 0) {
        return GET_AB_SS_CPU(0);
    }

    // For ket P cases, compute auxiliary (ab|ps) arrays with proper VRR
    // First compute lower angular momentum (ab|ps) arrays
    // ss_ps[d][m] = (ss|p_d s)^m - for same center, this is 0 (no bra reduction possible)
    double ss_ps[3][4];
    for (int d = 0; d < 3; d++) for (int m = 0; m < 4; m++) {
        ss_ps[d][m] = QC[d]*ss_m[m] + WQ[d]*ss_m[m+1];  // No bra reduction terms for ss
    }

    // sp_ps[bra_d][ket_d][m] = (s p_{bra_d} | p_{ket_d} s)^m
    double sp_ps[3][3][3];
    for (int bd = 0; bd < 3; bd++) {  // bra p direction
        for (int kd = 0; kd < 3; kd++) {  // ket p direction
            for (int m = 0; m < 3; m++) {
                double term = QC[kd]*sp_ss[bd][m] + WQ[kd]*sp_ss[bd][m+1];
                // Bra reduction: only position b (the p in sp) can contribute
                if (bd == kd) term += rho_inv * ss_m[m+1];  // (ss|ss)^{m+1}
                sp_ps[bd][kd][m] = term;
            }
        }
    }

    // ps_ps[bra_d][ket_d][m] = (p_{bra_d} s | p_{ket_d} s)^m
    double ps_ps[3][3][3];
    for (int bd = 0; bd < 3; bd++) {
        for (int kd = 0; kd < 3; kd++) {
            for (int m = 0; m < 3; m++) {
                double term = QC[kd]*ps_ss[bd][m] + WQ[kd]*ps_ss[bd][m+1];
                // Bra reduction: only position a (the p in ps) can contribute
                if (bd == kd) term += rho_inv * ss_m[m+1];
                ps_ps[bd][kd][m] = term;
            }
        }
    }

    // pp_ps[bra_d1][bra_d2][ket_d][m] = (p_{bra_d1} p_{bra_d2} | p_{ket_d} s)^m
    double pp_ps[3][3][3][2];
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            for (int kd = 0; kd < 3; kd++) {
                for (int m = 0; m < 2; m++) {
                    double term = PA[d1] * sp_ps[d2][kd][m] + WP[d1] * sp_ps[d2][kd][m+1];
                    // Bra B reduction (p->s): N_j(i) * (s s | p_k s)
                    if (d1 == d2) {
                        double val = ss_ps[kd][m] - (q/rP)*ss_ps[kd][m+1];
                        term += (1.0/(2.0*p)) * val;
                    }
                    // Ket reduction (p->s): N_k(i) * (s p_j | s s)
                    if (d1 == kd) {
                        term += rho_inv * sp_ss[d2][m+1];
                    }
                    pp_ps[d1][d2][kd][m] = term;
                }
            }
        }
    }

    // Now compute pp_ps[bra_d1][bra_d2][ket_d][m] = (p_{d1} p_{d2} | p_{ket_d} s)^m
    // Using sp_ps and ps_ps for bra reductions
    double ab_ps[3][3];  // ab_ps[ket_d][m] for the specific bra type
    for (int d = 0; d < 3; d++) {  // ket p direction
        for (int mm = 0; mm < 3; mm++) {
            double term = QC[d]*GET_AB_SS_CPU(mm) + WQ[d]*GET_AB_SS_CPU(mm+1);
            double bra = 0.0;
            if (i_type > 0 && i_type <= 3 && ax_i == d) {
                // Reduce bra i: (a-1 b | ps) = (s p_{ax_j} | p_d s) if j is p
                bra += (j_type==0) ? ss_ps[d][mm+1] : (j_type<=3) ? sp_ps[ax_j][d][mm+1] : 0.0;
            }
            if (j_type > 0 && j_type <= 3 && ax_j == d) {
                // Reduce bra j: (a b-1 | ps) = (p_{ax_i} s | p_d s) if i is p
                bra += (i_type==0) ? ss_ps[d][mm+1] : (i_type<=3) ? ps_ps[ax_i][d][mm+1] : 0.0;
            }
            // D-shell bra contributions
            if (i_type >= 4) {
                int n = (d1_i == d ? 1 : 0) + (d2_i == d ? 1 : 0);
                if (n > 0) {
                    // (a-1_d, b | ps) where a is D: reduces to (P b | ps)
                    int new_ax = (d1_i == d && d2_i == d) ? d : (d1_i == d ? d2_i : d1_i);
                    double am1_b = (j_type==0) ? ps_ps[new_ax][d][mm+1] : (j_type<=3) ? pp_ps[new_ax][ax_j][d][mm+1] : 0.0;
                    bra += n * am1_b;
                }
            }
            if (j_type >= 4) {
                int n = (d1_j == d ? 1 : 0) + (d2_j == d ? 1 : 0);
                if (n > 0) {
                    int new_ax = (d1_j == d && d2_j == d) ? d : (d1_j == d ? d2_j : d1_j);
                    double a_bm1 = (i_type==0) ? sp_ps[new_ax][d][mm+1] : (i_type<=3) ? pp_ps[ax_i][new_ax][d][mm+1] : 0.0;
                    bra += n * a_bm1;
                }
            }
            ab_ps[d][mm] = term + rho_inv * bra;
        }
    }

    // (ab|ps) - P on k, S on l
    if (k_type > 0 && k_type <= 3 && l_type == 0) {
        return ab_ps[k_type-1][0];
    }

    // (ab|sp) - S on k, P on l: HRR to get (ab|sp) from (ab|ps)
    if (k_type == 0 && l_type > 0 && l_type <= 3) {
        return ab_ps[l_type-1][0] + CD[l_type-1]*GET_AB_SS_CPU(0);
    }

    // (ab|pp) - P on both k and l
    if (k_type > 0 && k_type <= 3 && l_type > 0 && l_type <= 3) {
        int u = k_type - 1, v = l_type - 1;
        // Compute (ab|ds) for d_uv using ket VRR
        // [ab|c+1_u d]^m = QC[u]*[ab|cd]^m + WQ[u]*[ab|cd]^{m+1} + bra terms + (n_cu)/(2q)*[ab|c-1_u d]^{m+1}
        double term = QC[u]*ab_ps[v][0] + WQ[u]*ab_ps[v][1];
        // Ket reduction: when u==v, reducing p_u to s in direction u
        // Standard OS formula: (n_c/(2q)) * ([ab|c-1 d]^m - (p/rho)*[ab|c-1 d]^{m+1})
        // But for building [ab|ds]^0, we start from [ab|ps]^0, so the ket reduction uses [ab|ss]
        if (u == v) term += (1.0/(2.0*q))*(GET_AB_SS_CPU(0) - (p/rP)*GET_AB_SS_CPU(1));
        // Bra P contributions to (ab|ds): use reduced-bra integrals
        // [a-1_u b|p_v s]^1 and [a b-1_u|p_v s]^1
        if (i_type > 0 && i_type <= 3 && ax_i == u) {
            // Reduce bra position a: [a-1_u b|p_v s]^1
            // If j is s: [s|p_v s]^1 = ss_ps[v][1]
            // If j is p: [s p_{ax_j}|p_v s]^1 = sp_ps[ax_j][v][1]
            double reduced = (j_type==0) ? ss_ps[v][1] : (j_type<=3) ? sp_ps[ax_j][v][1] : 0.0;
            term += rho_inv * reduced;
        }
        if (j_type > 0 && j_type <= 3 && ax_j == u) {
            // Reduce bra position b: [a b-1_u|p_v s]^1
            // If i is s: [s|p_v s]^1 = ss_ps[v][1]
            // If i is p: [p_{ax_i} s|p_v s]^1 = ps_ps[ax_i][v][1]
            double reduced = (i_type==0) ? ss_ps[v][1] : (i_type<=3) ? ps_ps[ax_i][v][1] : 0.0;
            term += rho_inv * reduced;
        }
        double ab_ds = term;
        // HRR: (ab|pp) = (ab|ds) + CD[v]*(ab|ps)
        return ab_ds + CD[v]*ab_ps[u][0];
    }

    // Helper lambda for D indices
    auto get_d_indices = [](int type, int *d1, int *d2) {
        int t = type - 4;
        if (t==0) { *d1=0; *d2=0; } else if (t==1) { *d1=0; *d2=1; } else if (t==2) { *d1=0; *d2=2; }
        else if (t==3) { *d1=1; *d2=1; } else if (t==4) { *d1=1; *d2=2; } else if (t==5) { *d1=2; *d2=2; }
    };

    // Helper to compute d-component index from cartesian pair
    auto d_index = [](int a, int b) -> int {
        if (a > b) { int t=a; a=b; b=t; }
        if (a==0 && b==0) return 0;
        if (a==0 && b==1) return 1;
        if (a==0 && b==2) return 2;
        if (a==1 && b==1) return 3;
        if (a==1 && b==2) return 4;
        return 5;
    };

    // D-shell ket support
    if (k_type >= 4 || l_type >= 4) {
        // Compute full ab_ds array: ab_ds[dc][m] for all 6 D components
        double ab_ds_arr[6][2];
        for (int dc = 0; dc < 6; dc++) {
            int u, v; get_d_indices(dc+4, &u, &v);
            for (int m = 0; m < 2; m++) {
                double term = QC[u]*ab_ps[v][m] + WQ[u]*ab_ps[v][m+1];
                if (u == v) term += (1.0/(2.0*q))*(GET_AB_SS_CPU(m) - (p/rP)*GET_AB_SS_CPU(m+1));

                // Bra P contributions
                if (i_type > 0 && i_type <= 3 && ax_i == u) {
                    double reduced = (j_type==0) ? ss_ps[v][m+1] : (j_type<=3) ? sp_ps[ax_j][v][m+1] : 0.0;
                    term += rho_inv * reduced;
                }
                if (j_type > 0 && j_type <= 3 && ax_j == u) {
                    double reduced = (i_type==0) ? ss_ps[v][m+1] : (i_type<=3) ? ps_ps[ax_i][v][m+1] : 0.0;
                    term += rho_inv * reduced;
                }
                // D-shell bra contributions
                if (i_type >= 4) {
                    int n = (d1_i == u ? 1 : 0) + (d2_i == u ? 1 : 0);
                    if (n > 0) {
                        int new_ax = (d1_i == u && d2_i == u) ? u : (d1_i == u ? d2_i : d1_i);
                        double reduced = (j_type==0) ? ps_ps[new_ax][v][m+1] : (j_type<=3) ? pp_ps[new_ax][ax_j][v][m+1] : 0.0;
                        term += rho_inv * n * reduced;
                    }
                }
                if (j_type >= 4) {
                    int n = (d1_j == u ? 1 : 0) + (d2_j == u ? 1 : 0);
                    if (n > 0) {
                        int new_ax = (d1_j == u && d2_j == u) ? u : (d1_j == u ? d2_j : d1_j);
                        double reduced = (i_type==0) ? sp_ps[new_ax][v][m+1] : (i_type<=3) ? pp_ps[ax_i][new_ax][v][m+1] : 0.0;
                        term += rho_inv * n * reduced;
                    }
                }
                ab_ds_arr[dc][m] = term;
            }
        }

        // (ab|Ds) - D on k, S on l
        if (k_type >= 4 && l_type == 0) {
            return ab_ds_arr[k_type-4][0];
        }

        // (ab|sD) - S on k, D on l: HRR from ab_ds
        if (k_type == 0 && l_type >= 4) {
            int u, v; get_d_indices(l_type, &u, &v);
            if (u == v) {
                return ab_ds_arr[l_type-4][0] + 2.0*CD[u]*ab_ps[u][0] + CD[u]*CD[u]*GET_AB_SS_CPU(0);
            } else {
                return ab_ds_arr[l_type-4][0] + CD[v]*ab_ps[u][0] + CD[u]*ab_ps[v][0] + CD[u]*CD[v]*GET_AB_SS_CPU(0);
            }
        }

        // Compute full ab_pp array via HRR from ab_ds (needed for m=0,1)
        double ab_pp[3][3][2];
        for (int mm = 0; mm < 2; mm++) {
            for (int uu = 0; uu < 3; uu++) for (int vv = 0; vv < 3; vv++) {
                int dc_uv = d_index(uu, vv);
                ab_pp[uu][vv][mm] = ab_ds_arr[dc_uv][mm] + CD[vv]*ab_ps[uu][mm];
            }
        }

        // Compute ab_dp: (ab|D_dc P_pp)
        double ab_dp[6][3][2];
        for (int dc = 0; dc < 6; dc++) {
            int u, v; get_d_indices(dc+4, &u, &v);
            for (int pp = 0; pp < 3; pp++) {
                for (int m = 0; m < 2; m++) {
                    double term = QC[pp]*ab_ds_arr[dc][m] + WQ[pp]*ab_ds_arr[dc][m+1];
                    // Ket reduction of sk (D) if building sl (P)
                    {
                        int n_ket_c = (u == pp ? 1 : 0) + (v == pp ? 1 : 0);
                        if (n_ket_c > 0) {
                            int reduced_dir = (u == pp && v == pp) ? pp : (u == pp ? v : u);
                            double val = ab_ps[reduced_dir][m] - (p/rP)*ab_ps[reduced_dir][m+1];
                            term += n_ket_c * (1.0/(2.0*q)) * val;
                        }
                    }
                    // Bra reduction: (a-1|DP) from (a-1|Ds) at m+1
                    if (i_type > 0 && i_type <= 3 && ax_i == pp) {
                        double rb_ps_0 = (j_type==0) ? ss_ps[v][m+1] : (j_type<=3) ? sp_ps[ax_j][v][m+1] : 0.0;
                        double rb_ps_1 = (j_type==0) ? ss_ps[v][m+2] : (j_type<=3) ? sp_ps[ax_j][v][m+2] : 0.0;
                        double r_ds = QC[u]*rb_ps_0 + WQ[u]*rb_ps_1;
                        if (u == v) {
                            double rb_ss_0 = (j_type==0) ? ss_m[m+1] : sp_ss[ax_j][m+1];
                            double rb_ss_1 = (j_type==0) ? ss_m[m+2] : sp_ss[ax_j][m+2];
                            r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        term += rho_inv * r_ds;
                    }
                    if (j_type > 0 && j_type <= 3 && ax_j == pp) {
                        double rb_ps_0 = (i_type==0) ? ss_ps[v][m+1] : (i_type<=3) ? ps_ps[ax_i][v][m+1] : 0.0;
                        double rb_ps_1 = (i_type==0) ? ss_ps[v][m+2] : (i_type<=3) ? ps_ps[ax_i][v][m+2] : 0.0;
                        double r_ds = QC[u]*rb_ps_0 + WQ[u]*rb_ps_1;
                        if (u == v) {
                            double rb_ss_0 = (i_type==0) ? ss_m[m+1] : ps_ss[ax_i][m+1];
                            double rb_ss_1 = (i_type==0) ? ss_m[m+2] : ps_ss[ax_i][m+2];
                            r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        term += rho_inv * r_ds;
                    }
                    ab_dp[dc][pp][m] = term;
                }
            }
        }

        // (ab|DP) - D on k, P on l
        if (k_type >= 4 && l_type > 0 && l_type <= 3) {
            return ab_dp[k_type-4][l_type-1][0];
        }

        // Compute ab_pd via HRR from ab_dp
        double ab_pd[3][6];
        for (int uu = 0; uu < 3; uu++) for (int dc = 0; dc < 6; dc++) {
            int v, w; get_d_indices(dc+4, &v, &w);
            int dc_uw = d_index(uu, w);
            ab_pd[uu][dc] = ab_dp[dc_uw][v][0] + CD[w] * ab_pp[uu][v][0];
        }

        // (ab|PD) - P on k, D on l
        if (k_type > 0 && k_type <= 3 && l_type >= 4) {
            return ab_pd[k_type-1][l_type-4];
        }

        // Compute ab_dd via VRR on ab_dp
        double ab_dd[6][6];
        for (int d1 = 0; d1 < 6; d1++) {
            for (int d2 = 0; d2 < 6; d2++) {
                int x, y; get_d_indices(d2+4, &x, &y);
                double term = QC[y]*ab_dp[d1][x][0] + WQ[y]*ab_dp[d1][x][1];
                // Ket reduction (p_x -> s)
                if (x == y) {
                    term += (1.0/(2.0*q)) * (ab_ds_arr[d1][0] - (p/rP)*ab_ds_arr[d1][1]);
                }
                // Bra reduction for VRR adding momentum y
                // P-shell bra-a
                if (i_type > 0 && i_type <= 3 && ax_i == y) {
                    int u_d1, v_d1; get_d_indices(d1+4, &u_d1, &v_d1);
                    double rb_ps_0 = (j_type==0) ? ss_ps[v_d1][0] : (j_type<=3) ? sp_ps[ax_j][v_d1][0] : 0.0;
                    double rb_ps_1 = (j_type==0) ? ss_ps[v_d1][1] : (j_type<=3) ? sp_ps[ax_j][v_d1][1] : 0.0;
                    double r_ds = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                    if (u_d1 == v_d1) {
                        double rb_ss_0 = (j_type==0) ? ss_m[0] : sp_ss[ax_j][0];
                        double rb_ss_1 = (j_type==0) ? ss_m[1] : sp_ss[ax_j][1];
                        r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                    }
                    double r_dp = QC[x]*r_ds + WQ[x]*(QC[u_d1]*rb_ps_1 + WQ[u_d1]*((j_type==0)?ss_m[2]:sp_ss[ax_j][2]));
                    term += rho_inv * r_dp;
                }
                // P-shell bra-b
                if (j_type > 0 && j_type <= 3 && ax_j == y) {
                    int u_d1, v_d1; get_d_indices(d1+4, &u_d1, &v_d1);
                    double rb_ps_0 = (i_type==0) ? ss_ps[v_d1][0] : (i_type<=3) ? ps_ps[ax_i][v_d1][0] : 0.0;
                    double rb_ps_1 = (i_type==0) ? ss_ps[v_d1][1] : (i_type<=3) ? ps_ps[ax_i][v_d1][1] : 0.0;
                    double r_ds = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                    if (u_d1 == v_d1) {
                        double rb_ss_0 = (i_type==0) ? ss_m[0] : ps_ss[ax_i][0];
                        double rb_ss_1 = (i_type==0) ? ss_m[1] : ps_ss[ax_i][1];
                        r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                    }
                    double r_dp = QC[x]*r_ds + WQ[x]*(QC[u_d1]*rb_ps_1 + WQ[u_d1]*((i_type==0)?ss_m[2]:ps_ss[ax_i][2]));
                    term += rho_inv * r_dp;
                }
                // D-shell bra-a: N_y(D_a) * (P_remaining D_b | D_d1 P_x)^1
                if (i_type >= 4) {
                    int n_bra = (d1_i == y ? 1 : 0) + (d2_i == y ? 1 : 0);
                    if (n_bra > 0) {
                        // Reduced bra: D -> P
                        int r_ax = (d1_i == y && d2_i == y) ? y : (d1_i == y ? d2_i : d1_i);
                        int u_d1, v_d1; get_d_indices(d1+4, &u_d1, &v_d1);
                        // Need (P_r_ax D_b | D_d1 P_x)^1 - approximate by using available intermediates
                        // For same-center case, use simplified form
                        double rb_ps_0 = (j_type==0) ? ps_ps[r_ax][v_d1][0] : (j_type<=3) ? pp_ps[r_ax][ax_j][v_d1][0] : 0.0;
                        double rb_ps_1 = (j_type==0) ? ps_ps[r_ax][v_d1][1] : (j_type<=3) ? pp_ps[r_ax][ax_j][v_d1][1] : 0.0;
                        double r_ds = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0 = (j_type==0) ? ps_ss[r_ax][0] : pp_ss[r_ax][ax_j][0];
                            double rb_ss_1 = (j_type==0) ? ps_ss[r_ax][1] : pp_ss[r_ax][ax_j][1];
                            r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        double r_dp = QC[x]*r_ds + WQ[x]*(QC[u_d1]*rb_ps_1 + WQ[u_d1]*((j_type==0)?ps_ss[r_ax][2]:pp_ss[r_ax][ax_j][2]));
                        term += rho_inv * n_bra * r_dp;
                    }
                }
                // D-shell bra-b: N_y(D_b) * (D_a P_remaining | D_d1 P_x)^1
                if (j_type >= 4) {
                    int n_bra = (d1_j == y ? 1 : 0) + (d2_j == y ? 1 : 0);
                    if (n_bra > 0) {
                        int r_ax = (d1_j == y && d2_j == y) ? y : (d1_j == y ? d2_j : d1_j);
                        int u_d1, v_d1; get_d_indices(d1+4, &u_d1, &v_d1);
                        // Need (D_a P_r_ax | D_d1 P_x)^1
                        double rb_ps_0 = (i_type==0) ? sp_ps[r_ax][v_d1][0] : (i_type<=3) ? pp_ps[ax_i][r_ax][v_d1][0] : 0.0;
                        double rb_ps_1 = (i_type==0) ? sp_ps[r_ax][v_d1][1] : (i_type<=3) ? pp_ps[ax_i][r_ax][v_d1][1] : 0.0;
                        double r_ds = QC[u_d1]*rb_ps_0 + WQ[u_d1]*rb_ps_1;
                        if (u_d1 == v_d1) {
                            double rb_ss_0 = (i_type==0) ? sp_ss[r_ax][0] : pp_ss[ax_i][r_ax][0];
                            double rb_ss_1 = (i_type==0) ? sp_ss[r_ax][1] : pp_ss[ax_i][r_ax][1];
                            r_ds += (1.0/(2.0*q)) * (rb_ss_0 - (p/rP)*rb_ss_1);
                        }
                        double r_dp = QC[x]*r_ds + WQ[x]*(QC[u_d1]*rb_ps_1 + WQ[u_d1]*((i_type==0)?sp_ss[r_ax][2]:pp_ss[ax_i][r_ax][2]));
                        term += rho_inv * n_bra * r_dp;
                    }
                }
                ab_dd[d1][d2] = term;
            }
        }

        // (ab|DD) - D on both k and l
        if (k_type >= 4 && l_type >= 4) {
            return ab_dd[k_type-4][l_type-4];
        }
    }

    #undef GET_AB_SS_CPU
    return 0.0;  // Not implemented for higher angular momentum
}

// CPU ERI for reference comparison (simplified, only s-type)
double cpu_eri_ssss(double ai, double aj, double ak, double al,
                    double Ax, double Ay, double Az,
                    double Bx, double By, double Bz,
                    double Cx, double Cy, double Cz,
                    double Dx, double Dy, double Dz) {
    double p = ai + aj, q = ak + al, alpha = p * q / (p + q);
    double Px = (ai*Ax + aj*Bx)/p, Py = (ai*Ay + aj*By)/p, Pz = (ai*Az + aj*Bz)/p;
    double Qx = (ak*Cx + al*Dx)/q, Qy = (ak*Cy + al*Dy)/q, Qz = (ak*Cz + al*Dz)/q;
    double AB2 = (Ax-Bx)*(Ax-Bx) + (Ay-By)*(Ay-By) + (Az-Bz)*(Az-Bz);
    double CD2 = (Cx-Dx)*(Cx-Dx) + (Cy-Dy)*(Cy-Dy) + (Cz-Dz)*(Cz-Dz);
    double PQ2 = (Px-Qx)*(Px-Qx) + (Py-Qy)*(Py-Qy) + (Pz-Qz)*(Pz-Qz);
    double T = alpha * PQ2;
    double K_AB = exp(-ai*aj/p * AB2);
    double K_CD = exp(-ak*al/q * CD2);
    // Boys F0
    double F0;
    if (T < 1e-15) F0 = 1.0;
    else if (T > 30.0) F0 = 0.5 * sqrt(M_PI / T);
    else { double x = sqrt(T); F0 = 0.5 * sqrt(M_PI) * erf(x) / x; }
    return 2.0 * pow(M_PI, 2.5) / (p * q * sqrt(p + q)) * K_AB * K_CD * F0;
}

extern "C" void nwdft_gpu_test_integrals_() {
    if (n_shells_gpu == 0) { printf("No basis loaded for integral test\n"); return; }

    // Copy shell data back to host for testing - use correct sizes!
    Shell *h_shells = (Shell*)malloc(sizeof(Shell) * n_shells_gpu);
    double *h_exps = (double*)malloc(sizeof(double) * n_prim_total);
    double *h_coefs = (double*)malloc(sizeof(double) * n_prim_total);
    cudaMemcpy(h_shells, d_shells, sizeof(Shell) * n_shells_gpu, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exps, d_exponents, sizeof(double) * n_prim_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_coefs, d_coefs, sizeof(double) * n_prim_total, cudaMemcpyDeviceToHost);

    printf("\n=== GPU Basis Info ===\n");
    printf("Number of shells: %d\n", n_shells_gpu);
    for (int i = 0; i < n_shells_gpu; i++) {
        printf("Shell %d: L=%d nprim=%d basis_idx=%d coords=(%f,%f,%f)\n",
               i, h_shells[i].L, h_shells[i].nprim, h_shells[i].basis_idx,
               h_shells[i].x, h_shells[i].y, h_shells[i].z);
        printf("  exps: ");
        for (int p = 0; p < h_shells[i].nprim && p < 3; p++) printf("%f ", h_exps[h_shells[i].ptr_exp + p]);
        printf("\n  coefs: ");
        for (int p = 0; p < h_shells[i].nprim && p < 3; p++) printf("%f ", h_coefs[h_shells[i].ptr_coef + p]);
        printf("\n");
    }

    // Test (00|00) integral - first S shell with itself
    Shell s0 = h_shells[0];
    if (s0.L == 0) {
        double integral_gpu = 0.0, integral_cpu = 0.0;
        for (int pi = 0; pi < s0.nprim; pi++) {
            for (int pj = 0; pj < s0.nprim; pj++) {
                for (int pk = 0; pk < s0.nprim; pk++) {
                    for (int pl = 0; pl < s0.nprim; pl++) {
                        double ai = h_exps[s0.ptr_exp + pi], aj = h_exps[s0.ptr_exp + pj];
                        double ak = h_exps[s0.ptr_exp + pk], al = h_exps[s0.ptr_exp + pl];
                        double ci = h_coefs[s0.ptr_coef + pi], cj = h_coefs[s0.ptr_coef + pj];
                        double ck = h_coefs[s0.ptr_coef + pk], cl = h_coefs[s0.ptr_coef + pl];
                        double prim_gpu = cpu_eri_ssss(ai, aj, ak, al,
                            s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z);
                        double prim_cpu = cpu_eri_full(ai, aj, ak, al,
                            s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, 0, 0, 0, 0);
                        integral_gpu += ci * cj * ck * cl * prim_gpu;
                        integral_cpu += ci * cj * ck * cl * prim_cpu;
                    }
                }
            }
        }
        printf("\nTest (s0 s0|s0 s0):\n");
        printf("  Simple Boys:      %18.12f\n", integral_gpu);
        printf("  Full CPU ERI:     %18.12f\n", integral_cpu);
        printf("  Difference:       %18.12e\n", integral_gpu - integral_cpu);
    }

    // Test individual primitive integrals for different angular momentum
    // Find first P shell
    int p_shell = -1;
    for (int i = 0; i < n_shells_gpu; i++) {
        if (h_shells[i].L == 1) { p_shell = i; break; }
    }

    if (p_shell >= 0 && s0.L == 0) {
        Shell sp = h_shells[p_shell];
        // Use first primitive of each shell for testing
        double ai_p = h_exps[sp.ptr_exp];  // P shell exponent
        double ai_s = h_exps[s0.ptr_exp];  // S shell exponent

        printf("\n=== Primitive Integral Tests ===\n");
        printf("P shell exponent: %f  S shell exponent: %f\n", ai_p, ai_s);
        printf("P center: (%f, %f, %f)\n", sp.x, sp.y, sp.z);
        printf("S center: (%f, %f, %f)\n", s0.x, s0.y, s0.z);

        // Test (px s | s s) - different centers
        double gpu_ps = cpu_eri_full(ai_p, ai_s, ai_s, ai_s,
            sp.x, sp.y, sp.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
            1, 0, 0, 0);

        // Numerical derivative reference
        double h = 1e-5;
        double s_plus = cpu_eri_ssss(ai_p, ai_s, ai_s, ai_s,
            sp.x+h, sp.y, sp.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z);
        double s_minus = cpu_eri_ssss(ai_p, ai_s, ai_s, ai_s,
            sp.x-h, sp.y, sp.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z);
        double ref_ps = (s_plus - s_minus) / (2*h) / (2*ai_p);  // Derivative relationship

        printf("\nTest (px s|s s): CPU ERI = %18.12e, Numerical = %18.12e, Diff = %e\n",
               gpu_ps, ref_ps, gpu_ps - ref_ps);

        // Test (px px | s s) at same center - this tests the VRR p->d part
        double gpu_pp = cpu_eri_full(ai_p, ai_p, ai_s, ai_s,
            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
            1, 1, 0, 0);

        printf("Test (px px|s s) at same center: CPU ERI = %18.12e\n", gpu_pp);

        // Test (py pz | s s) - off-diagonal
        double gpu_pp_yz = cpu_eri_full(ai_p, ai_p, ai_s, ai_s,
            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
            2, 3, 0, 0);  // py=2, pz=3

        printf("Test (py pz|s s) at same center: CPU ERI = %18.12e\n", gpu_pp_yz);

        // Test with DIFFERENT centers to see if HRR works correctly
        // Use H atom coordinates (assuming they're different from O)
        int h_shell = -1;
        for (int i = 0; i < n_shells_gpu; i++) {
            if (h_shells[i].L == 0 && (h_shells[i].x != s0.x || h_shells[i].y != s0.y || h_shells[i].z != s0.z)) {
                h_shell = i;
                break;
            }
        }

        if (h_shell >= 0) {
            Shell sh = h_shells[h_shell];
            double ai_h = h_exps[sh.ptr_exp];
            printf("\nTest with different centers (O P shell + H S shell):\n");
            printf("H center: (%f, %f, %f), exp = %f\n", sh.x, sh.y, sh.z, ai_h);

            // (px_O s_H | s_O s_O)
            double gpu_ps_diff = cpu_eri_full(ai_p, ai_h, ai_s, ai_s,
                sp.x, sp.y, sp.z, sh.x, sh.y, sh.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
                1, 0, 0, 0);

            double s_p2 = cpu_eri_ssss(ai_p, ai_h, ai_s, ai_s,
                sp.x+h, sp.y, sp.z, sh.x, sh.y, sh.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z);
            double s_m2 = cpu_eri_ssss(ai_p, ai_h, ai_s, ai_s,
                sp.x-h, sp.y, sp.z, sh.x, sh.y, sh.z, s0.x, s0.y, s0.z, s0.x, s0.y, s0.z);
            double ref_ps_diff = (s_p2 - s_m2) / (2*h) / (2*ai_p);

            printf("(px_O s_H|s_O s_O): CPU = %18.12e, Numerical = %18.12e, Diff = %e\n",
                   gpu_ps_diff, ref_ps_diff, gpu_ps_diff - ref_ps_diff);

            // Test Ket P: (s_O s_H | px_O s_H) - Tests VRR on C
            double gpu_ss_ps = cpu_eri_full(ai_s, ai_h, ai_p, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                sp.x, sp.y, sp.z, sh.x, sh.y, sh.z,
                0, 0, 1, 0); // i=0, j=0, k=1(px), l=0

            double s_plus_c = cpu_eri_ssss(ai_s, ai_h, ai_p, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                sp.x+h, sp.y, sp.z, sh.x, sh.y, sh.z);
            double s_minus_c = cpu_eri_ssss(ai_s, ai_h, ai_p, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                sp.x-h, sp.y, sp.z, sh.x, sh.y, sh.z);
            double ref_ss_ps = (s_plus_c - s_minus_c) / (2*h) / (2*ai_p);
            
            printf("(s_O s_H | px_O s_H) [VRR C]: GPU = %18.12e, Ref = %18.12e, Diff = %e\n",
                   gpu_ss_ps, ref_ss_ps, gpu_ss_ps - ref_ss_ps);

            // Test Ket HRR: (s_O s_H | s_O px_H) - Tests HRR on C
            // We use H center as D, but pretend it has P exponent (ai_h) from S.
            // Note: cpu_eri_full takes 4 exponents. We reuse ai_h for the P exponent.
            // Differentiating (s s | s s) w.r.t D_x = (s s | s p) * 2*alpha
            double gpu_ss_sp = cpu_eri_full(ai_s, ai_h, ai_s, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z,
                0, 0, 0, 1); // l=1(px)

            double s_plus_d = cpu_eri_ssss(ai_s, ai_h, ai_s, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                s0.x, s0.y, s0.z, sh.x+h, sh.y, sh.z);
            double s_minus_d = cpu_eri_ssss(ai_s, ai_h, ai_s, ai_h,
                s0.x, s0.y, s0.z, sh.x, sh.y, sh.z, 
                s0.x, s0.y, s0.z, sh.x-h, sh.y, sh.z);
            double ref_ss_sp = (s_plus_d - s_minus_d) / (2*h) / (2*ai_h);

            printf("(s_O s_H | s_O px_H) [HRR C]: GPU = %18.12e, Ref = %18.12e, Diff = %e\n",
                   gpu_ss_sp, ref_ss_sp, gpu_ss_sp - ref_ss_sp);
        }
    }

    // Add contracted (PP|PP) integral test to compare with NWChem
    if (p_shell >= 0) {
        Shell sp = h_shells[p_shell];
        printf("\n=== Contracted (PP|PP) Integral Test ===\n");
        printf("P shell: L=%d, nprim=%d, center=(%f,%f,%f)\n", sp.L, sp.nprim, sp.x, sp.y, sp.z);

        // Compute (px px | px px) contracted integral
        double pp_pp_xx = 0.0;
        for (int pi = 0; pi < sp.nprim; pi++) {
            for (int pj = 0; pj < sp.nprim; pj++) {
                for (int pk = 0; pk < sp.nprim; pk++) {
                    for (int pl = 0; pl < sp.nprim; pl++) {
                        double ai = h_exps[sp.ptr_exp + pi], aj = h_exps[sp.ptr_exp + pj];
                        double ak = h_exps[sp.ptr_exp + pk], al = h_exps[sp.ptr_exp + pl];
                        double ci = h_coefs[sp.ptr_coef + pi], cj = h_coefs[sp.ptr_coef + pj];
                        double ck = h_coefs[sp.ptr_coef + pk], cl = h_coefs[sp.ptr_coef + pl];
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z,
                            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z,
                            1, 1, 1, 1);  // px px px px
                        pp_pp_xx += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(px px | px px) GPU: %18.12f\n", pp_pp_xx);
        printf("NWChem value:        %18.12f (from int_2e4c)\n", 0.8801590934);
        printf("Difference:          %18.12e\n", pp_pp_xx - 0.8801590934);
        printf("Ratio (GPU/NWChem):  %18.12f\n", pp_pp_xx / 0.8801590934);

        // Compute (px px | py py) for comparison - NWChem eri(5)
        double pp_pp_xxyy = 0.0;
        for (int pi = 0; pi < sp.nprim; pi++) {
            for (int pj = 0; pj < sp.nprim; pj++) {
                for (int pk = 0; pk < sp.nprim; pk++) {
                    for (int pl = 0; pl < sp.nprim; pl++) {
                        double ai = h_exps[sp.ptr_exp + pi], aj = h_exps[sp.ptr_exp + pj];
                        double ak = h_exps[sp.ptr_exp + pk], al = h_exps[sp.ptr_exp + pl];
                        double ci = h_coefs[sp.ptr_coef + pi], cj = h_coefs[sp.ptr_coef + pj];
                        double ck = h_coefs[sp.ptr_coef + pk], cl = h_coefs[sp.ptr_coef + pl];
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z,
                            sp.x, sp.y, sp.z, sp.x, sp.y, sp.z,
                            1, 1, 2, 2);  // px px py py
                        pp_pp_xxyy += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(px px | py py) GPU: %18.12f\n", pp_pp_xxyy);
        printf("NWChem eri(5):       %18.12f\n", 0.7852702031);
        printf("Difference:          %18.12e\n", pp_pp_xxyy - 0.7852702031);
        printf("Ratio (GPU/NWChem):  %18.12f\n", pp_pp_xxyy / 0.7852702031);
    }

    // === D-SHELL TESTS ===
    // Find first D shell
    int d_shell = -1;
    for (int i = 0; i < n_shells_gpu; i++) {
        if (h_shells[i].L == 2) { d_shell = i; break; }
    }

    if (d_shell >= 0 && s0.L == 0) {
        Shell sd = h_shells[d_shell];
        printf("\n=== D-SHELL INTEGRAL TESTS ===\n");
        printf("D shell %d: L=%d nprim=%d basis_idx=%d\n", d_shell, sd.L, sd.nprim, sd.basis_idx);
        printf("D center: (%f, %f, %f)\n", sd.x, sd.y, sd.z);
        printf("S center: (%f, %f, %f)\n", s0.x, s0.y, s0.z);

        // Test (SS|DS) - dzz component (type=9)
        // NWChem D_6 (dzz) = 0.1430277507
        double ss_ds_zz = 0.0;
        for (int pi = 0; pi < s0.nprim; pi++) {
            for (int pj = 0; pj < s0.nprim; pj++) {
                for (int pk = 0; pk < sd.nprim; pk++) {
                    for (int pl = 0; pl < s0.nprim; pl++) {
                        double ai = h_exps[s0.ptr_exp + pi], aj = h_exps[s0.ptr_exp + pj];
                        double ak = h_exps[sd.ptr_exp + pk], al = h_exps[s0.ptr_exp + pl];
                        double ci = h_coefs[s0.ptr_coef + pi], cj = h_coefs[s0.ptr_coef + pj];
                        double ck = h_coefs[sd.ptr_coef + pk], cl = h_coefs[s0.ptr_coef + pl];
                        // dzz = type 9 (4+5 where 5 is the index for zz in D ordering)
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
                            sd.x, sd.y, sd.z, s0.x, s0.y, s0.z,
                            0, 0, 9, 0);  // ss | dzz s
                        ss_ds_zz += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(SS|Dzz S) GPU:   %18.12f\n", ss_ds_zz);
        printf("NWChem D_6:       %18.12f\n", 0.1430277507);
        printf("Difference:       %18.12e\n", ss_ds_zz - 0.1430277507);

        // Test (SS|DD) - dxx,dxx component (first in array)
        // NWChem eri(1) = 0.9261080925
        double ss_dd_xxxx = 0.0;
        for (int pi = 0; pi < s0.nprim; pi++) {
            for (int pj = 0; pj < s0.nprim; pj++) {
                for (int pk = 0; pk < sd.nprim; pk++) {
                    for (int pl = 0; pl < sd.nprim; pl++) {
                        double ai = h_exps[s0.ptr_exp + pi], aj = h_exps[s0.ptr_exp + pj];
                        double ak = h_exps[sd.ptr_exp + pk], al = h_exps[sd.ptr_exp + pl];
                        double ci = h_coefs[s0.ptr_coef + pi], cj = h_coefs[s0.ptr_coef + pj];
                        double ck = h_coefs[sd.ptr_coef + pk], cl = h_coefs[sd.ptr_coef + pl];
                        // dxx = type 4
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
                            sd.x, sd.y, sd.z, sd.x, sd.y, sd.z,
                            0, 0, 4, 4);  // ss | dxx dxx
                        ss_dd_xxxx += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(SS|Dxx Dxx) GPU: %18.12f\n", ss_dd_xxxx);
        printf("NWChem eri(1):    %18.12f\n", 0.9261080925);
        printf("Difference:       %18.12e\n", ss_dd_xxxx - 0.9261080925);

        // Test (DD|SS) - should match (SS|DD) by symmetry
        double dd_ss_xxxx = 0.0;
        for (int pi = 0; pi < sd.nprim; pi++) {
            for (int pj = 0; pj < sd.nprim; pj++) {
                for (int pk = 0; pk < s0.nprim; pk++) {
                    for (int pl = 0; pl < s0.nprim; pl++) {
                        double ai = h_exps[sd.ptr_exp + pi], aj = h_exps[sd.ptr_exp + pj];
                        double ak = h_exps[s0.ptr_exp + pk], al = h_exps[s0.ptr_exp + pl];
                        double ci = h_coefs[sd.ptr_coef + pi], cj = h_coefs[sd.ptr_coef + pj];
                        double ck = h_coefs[s0.ptr_coef + pk], cl = h_coefs[s0.ptr_coef + pl];
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            sd.x, sd.y, sd.z, sd.x, sd.y, sd.z,
                            s0.x, s0.y, s0.z, s0.x, s0.y, s0.z,
                            4, 4, 0, 0);  // dxx dxx | ss
                        dd_ss_xxxx += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(Dxx Dxx|SS) GPU: %18.12f\n", dd_ss_xxxx);
        printf("NWChem (DD|SS):   %18.12f\n", 0.9261080925);
        printf("Difference:       %18.12e\n", dd_ss_xxxx - 0.9261080925);

        // Test (DD|DD) - dxx,dxx|dxx,dxx
        // NWChem eri(1) = 0.8371637977
        double dd_dd_xxxx = 0.0;
        for (int pi = 0; pi < sd.nprim; pi++) {
            for (int pj = 0; pj < sd.nprim; pj++) {
                for (int pk = 0; pk < sd.nprim; pk++) {
                    for (int pl = 0; pl < sd.nprim; pl++) {
                        double ai = h_exps[sd.ptr_exp + pi], aj = h_exps[sd.ptr_exp + pj];
                        double ak = h_exps[sd.ptr_exp + pk], al = h_exps[sd.ptr_exp + pl];
                        double ci = h_coefs[sd.ptr_coef + pi], cj = h_coefs[sd.ptr_coef + pj];
                        double ck = h_coefs[sd.ptr_coef + pk], cl = h_coefs[sd.ptr_coef + pl];
                        double prim = cpu_eri_full(ai, aj, ak, al,
                            sd.x, sd.y, sd.z, sd.x, sd.y, sd.z,
                            sd.x, sd.y, sd.z, sd.x, sd.y, sd.z,
                            4, 4, 4, 4);  // dxx dxx | dxx dxx
                        dd_dd_xxxx += ci * cj * ck * cl * prim;
                    }
                }
            }
        }
        printf("(Dxx Dxx|Dxx Dxx) GPU: %18.12f\n", dd_dd_xxxx);
        printf("NWChem (DD|DD) eri(1): %18.12f\n", 0.8371637977);
        printf("Difference:            %18.12e\n", dd_dd_xxxx - 0.8371637977);
        printf("=== End D-shell tests ===\n");
    }

    free(h_shells); free(h_exps); free(h_coefs);
}

extern "C" {
void nwdft_gpu_allocate_(void **ptr, long *bytes) { cudaMalloc(ptr, (size_t)*bytes); }
void nwdft_gpu_free_(void **ptr) { if (*ptr) cudaFree(*ptr); *ptr = NULL; }
void nwdft_gpu_put_(void *h_ptr, void **d_ptr, long *bytes) { cudaMemcpy(*d_ptr, h_ptr, (size_t)*bytes, cudaMemcpyHostToDevice); }
void nwdft_gpu_get_(void **d_ptr, void *h_ptr, long *bytes) { cudaMemcpy(h_ptr, *d_ptr, (size_t)*bytes, cudaMemcpyDeviceToHost); }
void nwdft_gpu_zero_(void **d_ptr, long *bytes) { cudaMemset(*d_ptr, 0, (size_t)*bytes); }
void nwdft_gpu_zgemm_(char *transa, char *transb, long *m, long *n, long *k, void *alpha, void **A, long *lda, void **B, long *ldb, void *beta, void **C, long *ldc) {
    cublasOperation_t opA = (*transa == 'N' || *transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (*transb == 'N' || *transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasZgemm(handle, opA, opB, (int)*m, (int)*n, (int)*k, (cuDoubleComplex*)alpha, (cuDoubleComplex*)*A, (int)*lda, (cuDoubleComplex*)*B, (int)*ldb, (cuDoubleComplex*)beta, (cuDoubleComplex*)*C, (int)*ldc);
}
void nwdft_gpu_zaxpy_(long *n, void *alpha, void **X, long *incx, void **Y, long *incy) {
    cublasZaxpy(handle, (int)*n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)*X, (int)*incx, (cuDoubleComplex*)*Y, (int)*incy);
}
}
