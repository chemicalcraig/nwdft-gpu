      subroutine rttddft_gpu_upload_ri(rtdb, basis_ao, basis_ri)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "bas.fh"
#include "global.fh"
#include "stdio.fh"
#include "tcgmsg.fh"
#include "util.fh"

      integer, intent(in) :: rtdb, basis_ao, basis_ri
      
      integer nbf_ao, nbf_ri
      integer l_3c, k_3c
      integer l_met, k_met
      integer l_eri, k_eri
      integer nsh_ao, nsh_ri
      integer nsh_ri_actual
      integer ish, jsh, ksh
      integer ndim_3c, ndim_met
      integer me
      integer l_scr, k_scr
      integer max_scr, max_buf
      integer nb, ns
      integer ish_ri, jsh_ri
      integer i_ri_lo, i_ri_hi, j_ri_lo, j_ri_hi, k_ri_lo, k_ri_hi
      integer ish_ao, jsh_ao
      integer i_ao_lo, i_ao_hi, j_ao_lo, j_ao_hi
      integer idx, i, j, k
      integer offset
      integer info
      integer l_ipiv, k_ipiv
      integer l_work, k_work
      integer basis_handles(2)
      
      me = ga_nodeid()

      ! Collective initialization of integrals
      call int_terminate()
      basis_handles(1) = basis_ao
      basis_handles(2) = basis_ri
      call int_init(rtdb, 2, basis_handles)
      call int_acc_std()

      if (me.eq.0) then
         if (.not. bas_numbf(basis_ao, nbf_ao))
     $        call errquit('rttddft_gpu_ri: bas_nbf',0, BASIS_ERR)
         if (.not. bas_numbf(basis_ri, nbf_ri))
     $        call errquit('rttddft_gpu_ri: bas_nbf',0, BASIS_ERR)

         write(luout,*) 'RTTDDFT GPU: Allocating RI memory (', 
     $        nbf_ao, 'x', nbf_ao, 'x', nbf_ri, ')...'
         call util_flush(luout)

         ndim_3c = nbf_ao * nbf_ao * nbf_ri
         ndim_met = nbf_ri * nbf_ri
         
         if (.not. ma_push_get(mt_dbl, ndim_3c, '3c_ints', 
     $        l_3c, k_3c)) then
            call errquit('rttddft_gpu_ri: no mem 3c', ndim_3c, MA_ERR)
         endif
            
         if (.not. ma_push_get(mt_dbl, ndim_met, 'inv_metric', 
     $        l_met, k_met)) then
            call errquit('rttddft_gpu_ri: no mem met', ndim_met, MA_ERR)
         endif

         write(luout,*) 'RTTDDFT GPU: Initializing integrals...'
         call util_flush(luout)

         call int_mem_2e3c(max_buf, max_scr)
         call int_mem_2e2c(nb, ns)
         max_buf = max(max_buf, nb)
         max_scr = max(max_scr, ns)
         
         if (.not. ma_push_get(mt_dbl, max_scr, 'int_scr', 
     $        l_scr, k_scr)) then
            call errquit('rttddft_gpu_ri: scr', max_scr, MA_ERR)
         endif
         if (.not. ma_push_get(mt_dbl, max_buf, 'int_eri', 
     $        l_eri, k_eri)) then
            call errquit('rttddft_gpu_ri: eri', max_buf, MA_ERR)
         endif

         write(luout,*) 'RTTDDFT GPU: Computing 2c metric...'
         call util_flush(luout)
         call dfill(ndim_met, 0.0d0, dbl_mb(k_met), 1)

         if (.not. bas_numcont(basis_ri, nsh_ri_actual)) 
     $        call errquit('bas_numcont',0, BASIS_ERR)

         do ish_ri = 1, nsh_ri_actual
            if (.not. bas_cn2bfr(basis_ri, ish_ri, i_ri_lo, 
     $           i_ri_hi))
     $           call errquit('bas_cn2bfr',0, BASIS_ERR)
            do jsh_ri = 1, nsh_ri_actual
               if (.not. bas_cn2bfr(basis_ri, jsh_ri, j_ri_lo, 
     $              j_ri_hi))
     $              call errquit('bas_cn2bfr',0, BASIS_ERR)
               call int_2e2c(basis_ri, ish_ri, basis_ri, jsh_ri,
     $              max_scr, dbl_mb(k_scr), max_buf, dbl_mb(k_eri)) 
               idx = 0
               do j = j_ri_lo, j_ri_hi
                  do i = i_ri_lo, i_ri_hi
                     offset = k_met + (j-1)*nbf_ri + (i-1)
                     dbl_mb(offset) = dbl_mb(k_eri+idx)
                     idx = idx + 1
                  end do
               end do
            end do
         end do

         write(luout,*) 'RTTDDFT GPU: Inverting 2c metric...'
         call util_flush(luout)
         if (.not. ma_push_get(mt_int, nbf_ri, 'ipiv', 
     $        l_ipiv, k_ipiv)) then
            call errquit('rttddft_gpu_ri: ipiv', nbf_ri, MA_ERR)
         endif
         if (.not. ma_push_get(mt_dbl, nbf_ri*nbf_ri, 'work', l_work, 
     $        k_work)) then
            call errquit('rttddft_gpu_ri: work', nbf_ri*nbf_ri, MA_ERR)
         endif
         call dgetrf(nbf_ri, nbf_ri, dbl_mb(k_met), nbf_ri, 
     $               int_mb(k_ipiv), info)
         if (info .ne. 0) call errquit('dgetrf failed', info, CALC_ERR)
         call dgetri(nbf_ri, dbl_mb(k_met), nbf_ri, int_mb(k_ipiv),
     $               dbl_mb(k_work), nbf_ri*nbf_ri, info)
         if (info .ne. 0) call errquit('dgetri failed', info, CALC_ERR)
         if (.not. ma_pop_stack(l_work)) call errquit('pop',0,MA_ERR)
         if (.not. ma_pop_stack(l_ipiv)) call errquit('pop',0,MA_ERR)

         write(luout,*) 'RTTDDFT GPU: Computing 3c integrals...'
         call util_flush(luout)
         write(luout,*) 'Handles: AO=', basis_ao, ' RI=', basis_ri
         call util_flush(luout)
         
         call dfill(ndim_3c, 0.0d0, dbl_mb(k_3c), 1)
         if (.not. bas_numcont(basis_ao, nsh_ao))
     $        call errquit('bas_numcont',0, BASIS_ERR)
         do ish_ao = 1, nsh_ao
            if (.not. bas_cn2bfr(basis_ao, ish_ao, i_ao_lo, 
     $           i_ao_hi))
     $           call errquit('bas_cn2bfr',0, BASIS_ERR)
            do jsh_ao = 1, nsh_ao
               if (.not. bas_cn2bfr(basis_ao, jsh_ao, j_ao_lo, 
     $              j_ao_hi))
     $              call errquit('bas_cn2bfr',0, BASIS_ERR)
               do ksh = 1, nsh_ri_actual
                  if (.not. bas_cn2bfr(basis_ri, ksh, 
     $                 k_ri_lo, k_ri_hi)) then
                     call errquit('bas_cn2bfr',0, BASIS_ERR)
                  endif
                  
                  if (ish_ao.eq.1 .and. jsh_ao.eq.1 .and. ksh.eq.1) then
                     write(luout,*) 'Call int_2e3c:', basis_ri, ksh, 
     $                    basis_ao, ish_ao, jsh_ao
                     call util_flush(luout)
                  endif

                  call int_2e3c(basis_ri, ksh, basis_ao, ish_ao, jsh_ao,
     $                 max_scr, dbl_mb(k_scr),
     $                 max_buf, dbl_mb(k_eri))
                  
                  idx = 0
                  do j = j_ao_lo, j_ao_hi
                     do i = i_ao_lo, i_ao_hi
                        do k = k_ri_lo, k_ri_hi
                           offset = k_3c + (i-1)
                           offset = offset + (j-1)*nbf_ao
                           offset = offset + (k-1)*nbf_ao*nbf_ao
                           dbl_mb(offset) = dbl_mb(k_eri + idx)
                           idx = idx + 1
                        end do
                     end do
                  end do
               end do
            end do
         end do

         if (.not. ma_pop_stack(l_eri)) call errquit('pop',0,MA_ERR)
         if (.not. ma_pop_stack(l_scr)) call errquit('pop',0,MA_ERR)
         
         call rttddft_gpu_upload_ri_data(nbf_ri, nbf_ao, 
     $        dbl_mb(k_3c), dbl_mb(k_met))

         if (.not. ma_pop_stack(l_met)) call errquit('pop',0,MA_ERR)
         if (.not. ma_pop_stack(l_3c)) call errquit('pop',0,MA_ERR)
      endif
      
      call ga_sync()

      end subroutine

      subroutine rttddft_gpu_j_ri(g_dens, g_j, nbf)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
#include "stdio.fh"

      integer, intent(in) :: g_dens, g_j, nbf
      integer l_p, k_p
      integer l_j, k_j
      integer me
      
      me = ga_nodeid()
      if (me.ne.0) return
      
      if (.not. ma_push_get(mt_dbl, nbf*nbf, 'h_p', l_p, k_p))
     $     call errquit('rttddft_gpu_j_ri: p', nbf*nbf, MA_ERR)
      if (.not. ma_push_get(mt_dbl, nbf*nbf, 'h_j', l_j, k_j))
     $     call errquit('rttddft_gpu_j_ri: j', nbf*nbf, MA_ERR)
           
      call ga_get(g_dens, 1, nbf, 1, nbf, dbl_mb(k_p), nbf)
      call rttddft_gpu_compute_j_ri(dbl_mb(k_p), dbl_mb(k_j), nbf)
      call ga_acc(g_j, 1, nbf, 1, nbf, dbl_mb(k_j), nbf, 1.0d0)
      
      if (.not. ma_pop_stack(l_j)) call errquit('pop',0,MA_ERR)
      if (.not. ma_pop_stack(l_p)) call errquit('pop',0,MA_ERR)
      
      end subroutine
