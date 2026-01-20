      subroutine dft_gpu_compute_k(g_dens, g_fock, nbf, ipol, factor)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
#include "stdio.fh"
#include "nwdft_gpu.fh"

      integer, intent(in) :: g_dens, g_fock
      integer, intent(in) :: nbf, ipol
      double precision, intent(in) :: factor
      
      integer l_P, k_P, l_J, k_J, l_K, k_K
      integer size, i
      integer me, nproc
      integer(8) :: nbf8, do_k8, me8, nproc8

      double precision val_dbl
      logical status_rtdb

      me = ga_nodeid()
      nproc = ga_nnodes()

      ! Currently only replicated data strategy (each node does full calculation if it has full density?)
      ! Actually, if g_dens is distributed, we need to gather it.
      ! The GPU kernel nwdft_gpu_compute_fock_jk expects full density matrix on host.
      ! This limits system size to single node memory.
      
      size = nbf * nbf
      if (.not. ma_push_get(mt_dbl, size, 'gpu_P', l_P, k_P))
     $     call errquit('dft_gpu_compute_k: ma P', size, MA_ERR)
      if (.not. ma_push_get(mt_dbl, size, 'gpu_J', l_J, k_J))
     $     call errquit('dft_gpu_compute_k: ma J', size, MA_ERR)
      if (.not. ma_push_get(mt_dbl, size, 'gpu_K', l_K, k_K))
     $     call errquit('dft_gpu_compute_k: ma K', size, MA_ERR)

      ! Prepare arguments for C wrapper (pass by reference)
      nbf8 = nbf
      do_k8 = 1
      me8 = me
      nproc8 = nproc

      if (me.eq.0) write(6,*) 'DEBUG: compute_k called'

      ! Loop over spin channels
      ! Note: If ipol=1, g_dens is Total Density? Standard DFT: yes.
      ! But K depends on alpha/beta density separately.
      ! For Closed Shell: K_tot = K(P_alpha) + K(P_beta) = K(0.5 P_tot) + K(0.5 P_tot) = 2 * K(0.5 P_tot)?
      ! Actually K[P] is linear. K[P_a + P_b] = K[P_a] + K[P_b].
      ! Wait, Exchange is same-spin only.
      ! K_alpha = K[P_alpha]. K_beta = K[P_beta].
      ! Total Energy Exchange = 0.5 * (P_alpha * K_alpha + P_beta * K_beta) ?
      ! Fock matrix F_alpha = H + J + K_alpha (HF).
      ! For RHF (ipol=1), P_tot = 2 * P_alpha. P_alpha = P_beta.
      ! F = H + 2J[P_alpha] - K[P_alpha].
      ! F = H + J[P_tot] - 0.5 * K[P_tot].
      ! The GPU kernel computes K[P] = Sum (mu nu | lambda sigma) P_lambda_sigma.
      ! So if we pass P_tot, we get K[P_tot].
      ! Then we need to scale it by 0.5 (or whatever kfac is).
      ! The scaling is handled by the caller (fock_2e or xc_getv) usually via kfac.
      ! xc_getv calls fock_2e with kfac = -0.5 * xfac for RHF.
      ! So we just need to return K[P_tot] and let caller scale it.
      
      ! Fetch density
      call ga_get(g_dens, 1, nbf, 1, nbf, dbl_mb(k_P), nbf)
      
      ! Call GPU Kernel
      ! We request K (do_k8=1). J is also computed but we might ignore it or use it?
      ! For now, we only care about K as per plan.
      
      call nwdft_gpu_compute_fock_jk(dbl_mb(k_P), dbl_mb(k_J), 
     $     dbl_mb(k_K), nbf8, do_k8, me8, nproc8)
      
      ! Accumulate K back to g_fock (Collective GA operation)
      call ga_acc(g_fock, 1, nbf, 1, nbf, dbl_mb(k_K), nbf, factor)
      
      call ga_sync()

      if (.not. ma_pop_stack(l_K)) call errquit('dft_gpu: pop K',0,0)
      if (.not. ma_pop_stack(l_J)) call errquit('dft_gpu: pop J',0,0)
      if (.not. ma_pop_stack(l_P)) call errquit('dft_gpu: pop P',0,0)

      end subroutine

      logical function dft_gpu_active()
      implicit none
#include "rtdb.fh"
#include "mafdecls.fh"
#include "global.fh"
      
      integer rtdb
      logical val
      external bgj_get_rtdb_handle
      integer bgj_get_rtdb_handle

      dft_gpu_active = .false.
      rtdb = bgj_get_rtdb_handle()
      
      if (rtdb_get(rtdb, 'dft:gpu', mt_log, 1, val)) then
         dft_gpu_active = val
      endif
      
      end function
