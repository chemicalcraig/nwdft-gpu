C
C     prop_magnus_exp_bch.F
C
C     Steps density matrix forward in time using Baker-Campbell-Hausdorff.
C
C     P(t+dt) = e^W P(t) e^-W
C             = P + 1/1! [W,P] + 1/2![W,[W,P]] + 1/3![W,[W,[W,P]]] + ...
C      
C     where W = -iF(t+dt/2).  Note both P and W are in MO basis, but we
C     drop the prime notation for clarity.  
C     
      subroutine prop_magnus_exp_bch(params, g_zscr1, g_zscr2,
     $     g_w, g_zdens)
      implicit none

#include "global.fh"
#include "errquit.fh"
#include "mafdecls.fh"
#include "stdio.fh"
#include "matutils.fh"
#include "rt_tddft.fh"

#ifdef RTTDDFT_CUDA
#include "rttddft_gpu.fh"
      integer*8 d_w, d_vn1, d_zscr1, d_zscr2, d_dens
      integer*8 mat_sz
      integer l_buf_w, k_buf_w
      integer l_buf_p, k_buf_p
      integer l_buf_vn1, k_buf_vn1
      logical gpu_active
      integer*8 n_c, inc_c
      complex*16 zalpha, zbeta
      double precision my_norm
#endif

      
C     == Inputs ==
      type(rt_params_t), intent(in) :: params
      integer, intent(in)           :: g_w               ! W_1 = -iF(t+dt/2) dt
      integer, intent(in)           :: g_zscr1, g_zscr2  ! work space


C     == In/out ==
      integer, intent(in) :: g_zdens           !P'(t) in; P'(t+dt) out


C     == Parameters ==
      character(*), parameter     :: pname = "prop_magnus_exp_bch: "
      integer, parameter          :: kmax = 2000 !max number of BCH terms


C     == Variables ==
      double complex invfac
      double complex zinvk
      logical converged
      integer ik
      integer g_vn1             ! commutator at k-1
      logical ok
      double precision norm
      integer me
      integer num_zeroterms
      integer n
      character*100 outstring
      double precision elapsed
      integer i

      if (params%prof) call prof_start (elapsed)


      me = ga_nodeid ()

      n = params%ns_mo

      ok = .true.
      ok = ok .and. ga_duplicate (g_zdens, g_vn1, "vn1")
      if (.not. ok) call errquit (pname//"alloc failed", 0, GA_ERR)

#ifdef RTTDDFT_CUDA
      gpu_active = .false.
      if (ga_nodeid().eq.0) gpu_active = .true.

      if (ga_nodeid().eq.0) then
         write(luout,*) 'RT-TDDFT GPU: Active = ', gpu_active
         write(luout,*) 'RT-TDDFT GPU: Matrix Size n = ', n
         call util_flush(luout)
      endif

      if (gpu_active) then
         n_c = n
         inc_c = 1
         mat_sz = n_c * n_c * 16 ! 16 bytes for double complex

         call rttddft_gpu_allocate(d_w, mat_sz)
         call rttddft_gpu_allocate(d_vn1, mat_sz)
         call rttddft_gpu_allocate(d_zscr1, mat_sz)
         call rttddft_gpu_allocate(d_zscr2, mat_sz)
         call rttddft_gpu_allocate(d_dens, mat_sz)

         if (.not. ma_push_get(mt_dcpl, n*n, 'buf_w', l_buf_w, k_buf_w))
     $        call errquit(pname//'ma failed w',0,MA_ERR)
         if (.not. ma_push_get(mt_dcpl, n*n, 'buf_p', l_buf_p, k_buf_p))
     $        call errquit(pname//'ma failed p',0,MA_ERR)
         if (.not. ma_push_get(mt_dcpl,n*n,'vn1',l_buf_vn1,k_buf_vn1))
     $        call errquit(pname//'ma failed vn1',0,MA_ERR)

         call ga_get(g_w, 1, n, 1, n, dcpl_mb(k_buf_w), n)
         call ga_get(g_zdens, 1, n, 1, n, dcpl_mb(k_buf_p), n)

         call rttddft_gpu_put(dcpl_mb(k_buf_w), d_w, mat_sz)
         call rttddft_gpu_put(dcpl_mb(k_buf_p), d_dens, mat_sz)
         
         ! k=0 term for vn1 is just P
         call rttddft_gpu_put(dcpl_mb(k_buf_p), d_vn1, mat_sz)
      endif
#endif

C
C     The k=0 term is just P.
C      
      ik = 0
      invfac = z1
      call ga_copy (g_zdens, g_vn1)

      
C
C     Do the summation from k=1 until kmax, or until converges.
C      
      num_zeroterms = 0
      converged = .false.
      
      do while (.not. converged)
#ifdef RTTDDFT_CUDA
         if (gpu_active) then
            ik = ik + 1
            zinvk = z1 / dcmplx (ik)
            
            ! zscr1 = zinvk * W * vn1
            zalpha = zinvk
            zbeta = z0
            call rttddft_gpu_zgemm('N', 'N', n_c, n_c, n_c, zalpha, 
     $           d_w, n_c, d_vn1, n_c, zbeta, d_zscr1, n_c)

            ! zscr2 = zinvk * vn1 * W
            call rttddft_gpu_zgemm('N', 'N', n_c, n_c, n_c, zalpha, 
     $           d_vn1, n_c, d_w, n_c, zbeta, d_zscr2, n_c)

            ! vn1 = zscr1 - zscr2
            call rttddft_gpu_zero(d_vn1, mat_sz)
            zalpha = z1
            call rttddft_gpu_zaxpy(n_c*n_c, zalpha, d_zscr1, inc_c, 
     $           d_vn1, inc_c)
            zalpha = zn1
            call rttddft_gpu_zaxpy(n_c*n_c, zalpha, d_zscr2, inc_c, 
     $           d_vn1, inc_c)

            ! P = P + vn1
            zalpha = z1
            call rttddft_gpu_zaxpy(n_c*n_c, zalpha, d_vn1, inc_c, 
     $           d_dens, inc_c)

            ! Check norm of vn1
            call rttddft_gpu_get(d_vn1, dcpl_mb(k_buf_vn1), mat_sz)
            
            my_norm = 0.0d0
            do i = 0, n*n-1
               if (abs(dcpl_mb(k_buf_vn1+i)) .gt. my_norm) then
                  my_norm = abs(dcpl_mb(k_buf_vn1+i))
               endif
            enddo
            
            if (my_norm .lt. params%tol_series)
     $           num_zeroterms = num_zeroterms + 1

            if (num_zeroterms .ge. params%terms_series)
     $           converged = .true.
            
            if (ik.gt.kmax)
     $           call errquit (pname//"failed to converge", 0, 0)

         endif
         
         ! Sync convergence status
         call ga_brdcst(mt_log, converged, 
     $        ma_sizeof(mt_log,1,mt_byte), 0)
         call ga_brdcst(mt_int, ik, 
     $        ma_sizeof(mt_int,1,mt_byte), 0)

#else
         call ga_sync ()

         ik = ik + 1
         zinvk = z1 / dcmplx (ik)
         
C     
C     Compute 1/k*[W, vn1], where vn1 is the commutator which includes the
C     1/k prefactor from the previous step.  Then take this result and
C     store it in vn1 for use in the next k step.
C
         call ga_zero (g_zscr1)
         call ga_zero (g_zscr2)
         
         call ga_zgemm ("N", "N", n, n, n,
     $        zinvk, g_w, g_vn1, z0, g_zscr1)
         call ga_zgemm ("N", "N", n, n, n,
     $        zinvk, g_vn1, g_w, z0, g_zscr2)

         call ga_zero (g_vn1)
         call ga_add (z1, g_zscr1, zn1, g_zscr2, g_vn1)

         
C
C     Accumulate 1/k [W, vn1] (computed above) into the P(t+dt) sum.  If
C     the density matrix has not changed the sum has converged and we
C     can stop.
C     
         norm = mat_norm (g_vn1)
         
         if (norm .lt. params%tol_series)
     $        num_zeroterms = num_zeroterms + 1

         if (num_zeroterms .ge. params%terms_series)
     $        converged = .true.
         
         call ga_add (z1, g_zdens, z1, g_vn1, g_zdens)
         norm = mat_norm (g_zdens)
         
C
C     Stop if we fail to converge after kmax terms.
C
         if (ik.gt.kmax)
     $        call errquit (pname//"failed to converge", 0, 0)
#endif
      enddo

      
C
C     Clean up.
C
#ifdef RTTDDFT_CUDA
      if (gpu_active) then
         call rttddft_gpu_get(d_dens, dcpl_mb(k_buf_p), mat_sz)
         ! Put back into GA
         call ga_put(g_zdens, 1, n, 1, n, dcpl_mb(k_buf_p), n)
         
         call rttddft_gpu_free(d_w)
         call rttddft_gpu_free(d_vn1)
         call rttddft_gpu_free(d_zscr1)
         call rttddft_gpu_free(d_zscr2)
         call rttddft_gpu_free(d_dens)
         
         if (.not. ma_pop_stack(l_buf_vn1))
     $        call errquit(pname//'pop failed',0,MA_ERR)
         if (.not. ma_pop_stack(l_buf_p))
     $        call errquit(pname//'pop failed',0,MA_ERR)
         if (.not. ma_pop_stack(l_buf_w))
     $        call errquit(pname//'pop failed',0,MA_ERR)
      endif
      call ga_sync()
#endif

      ok = .true.
      ok = ok .and. ga_destroy (g_vn1)
      if (.not.ok) call errquit (pname//"failed to destroy", 0, GA_ERR)


C
C     If profiling is enabled print convergence data.
C      
      if (params%prof) then 
         write(outstring,"(a,i0,a)") "BCH converged after ",ik," terms;"
         call prof_end (elapsed, trim(outstring))
      endif
      
      end subroutine

      
c $Id$