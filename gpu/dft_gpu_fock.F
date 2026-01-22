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
      integer size, i, j
      integer me, nproc
      integer(8) :: nbf8, do_k8, me8, nproc8

      double precision val_dbl, p_sum, p_trace, k_sum
      double precision effective_factor
      logical status_rtdb

      integer, save :: call_count = 0

      me = ga_nodeid()
      nproc = ga_nnodes()

      call_count = call_count + 1

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

      ! Debug: Print density matrix info on first call
      if (me.eq.0 .and. call_count.eq.1) then
         p_sum = 0.0d0
         p_trace = 0.0d0
         do i = 1, nbf
            do j = 1, nbf
               p_sum = p_sum + dbl_mb(k_P + (i-1)*nbf + (j-1))
            enddo
            p_trace = p_trace + dbl_mb(k_P + (i-1)*nbf + (i-1))
         enddo
         write(6,'(A,E18.10,A,E18.10)')
     $        ' Density matrix: sum=', p_sum, ' trace=', p_trace
         write(6,*) 'Density P[0:3,0:3]:'
         do i = 1, min(4, nbf)
            write(6,'(4F12.6)') dbl_mb(k_P+(i-1)*nbf),
     $           dbl_mb(k_P+(i-1)*nbf+1), dbl_mb(k_P+(i-1)*nbf+2),
     $           dbl_mb(k_P+(i-1)*nbf+3)
         enddo
         write(6,*) 'factor for K accumulation:', factor
         ! Call integral test
         call nwdft_gpu_test_integrals()
      endif

      ! Call GPU Kernel
      ! We request K (do_k8=1). J is also computed but we might ignore it or use it?
      ! For now, we only care about K as per plan.

      call nwdft_gpu_compute_fock_jk(dbl_mb(k_P), dbl_mb(k_J),
     $     dbl_mb(k_K), nbf8, do_k8, me8, nproc8)

      ! Debug: Print K matrix info on first call
      if (me.eq.0 .and. call_count.eq.1) then
         k_sum = 0.0d0
         do i = 1, nbf
            do j = 1, nbf
               k_sum = k_sum + dbl_mb(k_K + (i-1)*nbf + (j-1))
            enddo
         enddo
         write(6,'(A,E18.10)') ' Fortran K matrix sum=', k_sum

         ! Print GPU K matrix diagonal and first row
         write(6,*) '=== GPU K matrix (raw, unscaled) ==='
         write(6,*) 'Diagonal elements K[i,i]:'
         do i = 1, min(7, nbf)
            write(6,'(A,I2,A,F14.8)') '  K[',i,']  = ',
     $           dbl_mb(k_K + (i-1)*nbf + (i-1))
         enddo
         write(6,*) 'First row K[1,:]:'
         write(6,'(7F11.6)') (dbl_mb(k_K + j - 1), j=1, min(7,nbf))
      endif

      ! Accumulate K back to g_fock (Collective GA operation)
      ! No factor correction - use input factor directly
      if (me.eq.0 .and. call_count.eq.1) then
         write(6,'(A,F12.8)') ' GPU K factor=', factor
      endif
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

c     Test subroutine to compare NWChem int_2e4c with GPU integrals
      subroutine dft_gpu_test_int2e(basis, geom)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "bas.fh"
#include "global.fh"
#include "stdio.fh"

      integer, intent(in) :: basis, geom

      integer nshells, ish, jsh, ksh, lsh
      integer type_i, type_j, type_k, type_l
      integer nprim, ngen, sphcart
      integer lscr, leri
      integer l_scr, k_scr, l_eri, k_eri
      double precision scr(100000), eri(100000)
      integer i, j, k, l, idx
      integer ni, nj, nk, nl

      if (ga_nodeid().ne.0) return

      write(6,*) '=== NWChem int_2e4c comparison test ==='

      if (.not. bas_numcont(basis, nshells))
     $     call errquit('test_int2e: bas_numcont',0,BASIS_ERR)

      write(6,'(A,I4)') ' Number of shells: ', nshells

      lscr = 10000
      leri = 1000

c     Use our allocated scratch sizes directly
      lscr = 100000
      leri = 100000

c     Test a specific shell quartet: (P P | S S) using shells 3,3,1,1
c     Shell 1 is O 1s (S), Shell 3 is O 2p (P)
      ish = 3  ! P shell (if it exists)
      jsh = 3  ! P shell
      ksh = 1  ! S shell
      lsh = 1  ! S shell

      if (ish.gt.nshells .or. jsh.gt.nshells .or.
     $    ksh.gt.nshells .or. lsh.gt.nshells) then
         write(6,*) 'Shell indices out of range, using 1,1,1,1'
         ish = 1; jsh = 1; ksh = 1; lsh = 1
      endif

c     Get shell info
      if (.not. bas_continfo(basis, ish, type_i, nprim, ngen, sphcart))
     $     call errquit('test: continfo',0,BASIS_ERR)
      ni = 1
      if (type_i.eq.1) ni = 3
      if (type_i.eq.2) ni = 6
      if (type_i.eq.-1) ni = 4  ! SP shell

      if (.not. bas_continfo(basis, jsh, type_j, nprim, ngen, sphcart))
     $     call errquit('test: continfo',0,BASIS_ERR)
      nj = 1
      if (type_j.eq.1) nj = 3
      if (type_j.eq.2) nj = 6
      if (type_j.eq.-1) nj = 4

      if (.not. bas_continfo(basis, ksh, type_k, nprim, ngen, sphcart))
     $     call errquit('test: continfo',0,BASIS_ERR)
      nk = 1
      if (type_k.eq.1) nk = 3
      if (type_k.eq.2) nk = 6
      if (type_k.eq.-1) nk = 4

      if (.not. bas_continfo(basis, lsh, type_l, nprim, ngen, sphcart))
     $     call errquit('test: continfo',0,BASIS_ERR)
      nl = 1
      if (type_l.eq.1) nl = 3
      if (type_l.eq.2) nl = 6
      if (type_l.eq.-1) nl = 4

      write(6,'(A,4I3,A,4I3)') ' Shell quartet: ',ish,jsh,ksh,lsh,
     $     '  Types: ',type_i,type_j,type_k,type_l
      write(6,'(A,4I3)') ' Nbf per shell: ', ni, nj, nk, nl

c     Compute integrals using NWChem
      call int_2e4c(basis, ish, jsh, basis, ksh, lsh,
     $              lscr, scr, leri, eri)

      write(6,*) 'NWChem (ish jsh | ksh lsh) integrals:'
      idx = 0
      do i = 1, ni
         do j = 1, nj
            do k = 1, nk
               do l = 1, nl
                  idx = idx + 1
                  if (idx.le.20) then
                     write(6,'(A,I2,A,I2,A,I2,A,I2,A,E18.10)')
     $                    '  (',i,',',j,'|',k,',',l,') = ', eri(idx)
                  endif
               enddo
            enddo
         enddo
      enddo

      write(6,*) '=== End int_2e4c test ==='

c     Also test (PP|PP) shell quartet
      ish = 3  ! P shell
      jsh = 3
      ksh = 3
      lsh = 3

      if (ish.le.nshells) then
         call int_2e4c(basis, ish, jsh, basis, ksh, lsh,
     $                 lscr, scr, leri, eri)
         write(6,*) 'NWChem (PP|PP) integrals - first 10:'
         do idx = 1, 10
            write(6,'(A,I2,A,E18.10)') '  eri(',idx,') = ', eri(idx)
         enddo
         write(6,'(A,E18.10)') '  (px,px|px,px) eri(1) = ', eri(1)
c        For P shell ordering [px,py,pz], the (px px | px px) should be first
c        Storage: l varies fastest, then k, then j, then i
c        So eri(1) = (px px | px px), eri(28) = (py py | py py), etc.
         write(6,'(A,E18.10)') '  eri(28) = ', eri(28)
         write(6,'(A,E18.10)') '  eri(55) = ', eri(55)
      endif

c     Find D-shell for testing D-shell integrals
      do ish = 1, nshells
         if (.not. bas_continfo(basis, ish, type_i, nprim, ngen,
     $        sphcart)) call errquit('test: continfo',0,BASIS_ERR)
         if (type_i .eq. 2) then
            write(6,*) '=== D-SHELL INTEGRAL TESTS ==='
            write(6,'(A,I3)') ' Found D-shell at shell ', ish

c           Test (SS|DS) - S on bra, D-S on ket
            jsh = 1
            ksh = ish
            lsh = 1
            if (.not. bas_continfo(basis,jsh,type_j,nprim,ngen,
     $           sphcart)) call errquit('test',0,BASIS_ERR)
            if (.not. bas_continfo(basis,ksh,type_k,nprim,ngen,
     $           sphcart)) call errquit('test',0,BASIS_ERR)
            if (.not. bas_continfo(basis,lsh,type_l,nprim,ngen,
     $           sphcart)) call errquit('test',0,BASIS_ERR)
            if (type_j.eq.0 .and. type_l.eq.0) then
               call int_2e4c(basis, jsh, jsh, basis, ksh, lsh,
     $                       lscr, scr, leri, eri)
               write(6,*) 'NWChem (SS|DS) integrals:'
               do idx = 1, 6
                  write(6,'(A,I2,A,E18.10)') '  D_',idx,' = ', eri(idx)
               enddo
            endif

c           Test (SS|SD) - S on bra, S-D on ket
            ksh = 1   ! S shell
            lsh = ish ! D shell
            call int_2e4c(basis, jsh, jsh, basis, ksh, lsh,
     $                    lscr, scr, leri, eri)
            write(6,*) 'NWChem (SS|SD) integrals:'
            do idx = 1, 6
               write(6,'(A,I2,A,E18.10)') '  D_',idx,' = ', eri(idx)
            enddo

c           Test (SS|DD) - S on bra, D-D on ket
            ksh = ish ! D shell
            lsh = ish ! D shell
            call int_2e4c(basis, jsh, jsh, basis, ksh, lsh,
     $                    lscr, scr, leri, eri)
            write(6,*) 'NWChem (SS|DD) integrals (first 10):'
            do idx = 1, 10
               write(6,'(A,I2,A,E18.10)') '  eri(',idx,') = ',eri(idx)
            enddo

c           Test (PP|DS) - P on bra, D-S on ket
            do jsh = 1, nshells
               if (.not. bas_continfo(basis,jsh,type_j,nprim,ngen,
     $              sphcart)) call errquit('test',0,BASIS_ERR)
               if (type_j .eq. 1) then
                  ksh = ish ! D shell
                  lsh = 1   ! S shell
                  call int_2e4c(basis, jsh, jsh, basis, ksh, lsh,
     $                          lscr, scr, leri, eri)
                  write(6,*) 'NWChem (PP|DS) integrals (first 10):'
                  do idx = 1, 10
                     write(6,'(A,I2,A,E18.10)')'  eri(',idx,')= ',
     $                    eri(idx)
                  enddo
                  goto 100  ! exit after first P shell
               endif
            enddo
 100        continue

c           Test (DD|SS) - D on bra, S on ket
            ksh = 1   ! S shell
            lsh = 1   ! S shell
            call int_2e4c(basis, ish, ish, basis, ksh, lsh,
     $                    lscr, scr, leri, eri)
            write(6,*) 'NWChem (DD|SS) integrals (first 10):'
            do idx = 1, 10
               write(6,'(A,I2,A,E18.10)') '  eri(',idx,') = ', eri(idx)
            enddo

c           Test (DD|DD) - D on bra, D on ket
            ksh = ish ! D shell
            lsh = ish ! D shell
            call int_2e4c(basis, ish, ish, basis, ksh, lsh,
     $                    lscr, scr, leri, eri)
            write(6,*) 'NWChem (DD|DD) integrals (first 10):'
            do idx = 1, 10
               write(6,'(A,I2,A,E18.10)') '  eri(',idx,') = ', eri(idx)
            enddo

            write(6,*) '=== End D-shell tests ==='
            goto 200  ! exit after first D shell
         endif
      enddo
 200  continue

      end subroutine
