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

static cublasHandle_t handle = NULL;
static int is_init = 0;

extern "C" {

void nwdft_gpu_init_() {
    if (!is_init) {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            return;
        }
        cudaDeviceSetLimit(cudaLimitStackSize, 16384);
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
    int ax_i = (i_type>0 && i_type<=3) ? i_type-1 : -1, ax_j = (j_type>0 && j_type<=3) ? j_type-1 : -1;
    int d1_i=0, d2_i=0, d1_j=0, d2_j=0; if (i_type>=4) nwdft_get_d_indices(i_type, &d1_i, &d2_i); if (j_type>=4) nwdft_get_d_indices(j_type, &d1_j, &d2_j);
    #define GET_AB_SS(m) ( (i_type==0 && j_type==0) ? ss_m[m] : (i_type>0 && i_type<=3 && j_type==0) ? ps_ss[ax_i][m] : (i_type==0 && j_type>0 && j_type<=3) ? sp_ss[ax_j][m] : (i_type>0 && i_type<=3 && j_type>0 && j_type<=3) ? pp_ss[ax_i][ax_j][m] : (i_type>=4 && j_type==0) ? ds_ss[d1_i][d2_i][m] : (i_type==0 && j_type>=4) ? sd_ss[d1_j][d2_j][m] : 0.0 )
    #define GET_AM1_B_SS(m) ( (j_type==0) ? ss_m[m] : (j_type<=3) ? sp_ss[ax_j][m] : sd_ss[d1_j][d2_j][m] )
    #define GET_A_BM1_SS(m) ( (i_type==0) ? ss_m[m] : (i_type<=3) ? ps_ss[ax_i][m] : ds_ss[d1_i][d2_i][m] )
    double ab_ps[3][3]; for (int d=0; d<3; d++) for (int m=0; m<3; m++) {
        double term = QC[d]*GET_AB_SS(m) + WQ[d]*GET_AB_SS(m+1);
        double bra = 0.0;
        if (i_type > 0 && i_type <= 3 && ax_i == d) bra += GET_AM1_B_SS(m+1);
        if (j_type > 0 && j_type <= 3 && ax_j == d) bra += GET_A_BM1_SS(m+1);
        ab_ps[d][m] = term + rho_inv * bra;
    }
    if (k_type == 0 && l_type == 0) return GET_AB_SS(0);
    if (k_type <= 3 && l_type == 0) return ab_ps[k_type-1][0];
    if (k_type == 0 && l_type <= 3) return ab_ps[l_type-1][0] + CD[l_type-1]*GET_AB_SS(0);
    int d1_k=0, d2_k=0, d1_l=0, d2_l=0; if (k_type>=4) nwdft_get_d_indices(k_type, &d1_k, &d2_k); if (l_type>=4) nwdft_get_d_indices(l_type, &d1_l, &d2_l);
    int u = (k_type>=4)?d1_k:(k_type>0?k_type-1:0), v = (l_type>=4)?d1_l:(l_type>0?l_type-1:0);
    if (k_type>=4) { u=d1_k; v=d2_k; } else if (l_type>=4) { u=d1_l; v=d2_l; } else { u=k_type-1; v=l_type-1; }
    #define GET_AB_DS(uu, vv, m) ( QC[uu]*ab_ps[vv][m] + WQ[uu]*ab_ps[vv][m+1] + ((uu==vv)?(1.0/(2.0*q))*(GET_AB_SS(m)-(p/rP)*GET_AB_SS(m+1)):0.0) + rho_inv*((i_type>0&&i_type<=3&&ax_i==uu)?ab_ps[vv][m+1]:0.0) + rho_inv*((j_type>0&&j_type<=3&&ax_j==uu)?ab_ps[vv][m+1]:0.0) )
    if (k_type>=4 || l_type>=4) return GET_AB_DS(u, v, 0);
    return GET_AB_DS(u, v, 0) + CD[v]*ab_ps[u][0];
}

__global__ void nwdft_compute_K_kernel(int nshells, Shell *shells, double *exps, double *coefs, double *P, double *K, int nbf) {
    int ish = blockIdx.x * blockDim.x + threadIdx.x;
    int jsh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ish >= nshells || jsh >= nshells) return;
    
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
                    nwdft_atomicAdd_safe(&K[idx_k*nbf + idx_i], 1.0 * integral * P[idx_l*nbf + idx_j]);
                }
            }
        }
    }
}

extern "C" void nwdft_gpu_compute_fock_jk_(double *h_P, double *h_J, double *h_K, long *nbf, long *do_k) {
    if (!is_init) nwdft_gpu_init_();
    if (n_shells_gpu == 0) return;
    size_t size = sizeof(double) * (*nbf) * (*nbf);
    double *d_P, *d_K = NULL;
    cudaMalloc(&d_P, size); if (*do_k) cudaMalloc(&d_K, size);
    cudaMemcpy(d_P, h_P, size, cudaMemcpyHostToDevice);
    if (*do_k) cudaMemset(d_K, 0, size);
    
    dim3 threads(8, 8);
    dim3 blocks((n_shells_gpu + 7)/8, (n_shells_gpu + 7)/8);
    nwdft_compute_K_kernel<<<blocks, threads>>>(n_shells_gpu, d_shells, d_exponents, d_coefs, d_P, d_K, (int)*nbf);
    cudaDeviceSynchronize();
    if (*do_k) cudaMemcpy(h_K, d_K, size, cudaMemcpyDeviceToHost);
    cudaFree(d_P); if (d_K) cudaFree(d_K);
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
