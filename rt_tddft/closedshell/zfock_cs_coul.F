C
C     zfock_cs_coul.F
C
C     Calculates 2 electron part of a closed shell complex Fock matrix.
C     This routine is used when you *only* need to compute the Coulomb
C     term (i.e., pure DFT, with no Hartree-Fock exchange).  Note that
C     input is the real part of the density matrix, and output is real
C     part of the Fock matrix.
C
      subroutine zfock_cs_coul (params, g_densre, Ecoul, g_fc)
      implicit none

#include "bas.fh"
#include "errquit.fh"
#include "mafdecls.fh"
#include "stdio.fh"
#include "global.fh"
#include "util.fh"
#include "cdft.fh"
#include "rt_tddft.fh"

      
C     == Inputs ==
      type(rt_params_t), intent(in) :: params
      integer, intent(in)           :: g_densre

      
C     == Outputs ==
      double precision, intent(out) :: Ecoul
      integer, intent(in)           :: g_fc

      
C     == Parameters ==
      character(*), parameter     :: pname = "zfock_cs_coul:"
      

C     == Variables ==
      double precision jfac, kfac
C      double precision tol2e
      double precision elapsed

#ifdef RTTDDFT_CUDA
#include "rttddft_gpu.fh"
      integer l_p, k_p, l_j, k_j
      integer n
      integer do_k
#endif

      call rt_tddft_cs_confirm (params,'zfock_cs_coul.F')

      if (params%prof) call prof_start (elapsed)

      
C     == Initializations ==
C      tol2e=10.d0**(-itol2e)
      Ecoul = 0.d0

      call ga_zero (g_fc)

      jfac = 1d0
      kfac = 0d0
      

C     == Calculate Coulomb part of F and energy ==
#ifdef RTTDDFT_CUDA
      
      ! GPU Path: Rank 0 computes J matrix and broadcasts via GA
      
      if (ga_nodeid().eq.0) then
         n = params%nbf_ao
         
         if (params%iVcoul_opt .eq. 1) then
             call rttddft_gpu_j_ri(g_densre, g_fc, n)
         else
             if (.not. ma_push_get(mt_dbl, n*n, 'dens_gpu', l_p, k_p))
     $            call errquit(pname//'ma p',0,MA_ERR)
             if (.not. ma_push_get(mt_dbl, n*n, 'j_gpu', l_j, k_j))
     $            call errquit(pname//'ma j',0,MA_ERR)

             call ga_get(g_densre, 1, n, 1, n, dbl_mb(k_p), n)
         
             do_k = 0
             call rttddft_gpu_compute_fock_jk(dbl_mb(k_p), dbl_mb(k_j), 
     $            dbl_mb(k_j), n, do_k) ! K ptr is dummy if do_k=0
         
             call ga_put(g_fc, 1, n, 1, n, dbl_mb(k_j), n)

             if (.not. ma_pop_stack(l_j))
     $            call errquit(pname//'pop j',0,MA_ERR)
             if (.not. ma_pop_stack(l_p))
     $            call errquit(pname//'pop p',0,MA_ERR)
         endif
      endif
      
      ! All ranks wait for Rank 0 to update g_fc
      call ga_sync()
      
      ! Compute energy (parallel)
      Ecoul = 0.5d0 * ga_ddot (g_densre, g_fc)
      
      if (params%prof)
     $     call prof_end (elapsed,"Fock CS standard coul GPU")
      return

#else
      ! CPU Path: Standard NWChem fock_2e
      
      call fock_2e (geom, ao_bas_han, 1, jfac, kfac,
     $     params%tol2e_re, oskel, g_densre, g_fc, .false.)

      Ecoul = 0.5d0 * ga_ddot (g_densre, g_fc)
#endif


      if (params%prof) call prof_end (elapsed,"Fock CS standard coul")
      end subroutine
c $Id$
