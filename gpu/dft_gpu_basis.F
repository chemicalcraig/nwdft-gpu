      subroutine dft_gpu_upload_basis(basis, geom)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "bas.fh"
#include "geom.fh"
#include "global.fh"
#include "stdio.fh"

      integer, intent(in) :: basis, geom

      integer nshells, nshells_gpu
      integer ishell, type, nprim, ngeneral, sphcart
      integer iatom
      double precision charge
      character*16 tag
      
      integer, allocatable :: ang(:)
      integer, allocatable :: nprim_per_shell(:)
      integer, allocatable :: basis_idx(:)
      double precision, allocatable :: coords(:,:)
      double precision, allocatable :: exps_all(:)
      double precision, allocatable :: coefs_all(:)
      
      double precision, allocatable :: buf_exps(:), buf_coefs(:)
      
      integer nprim_total, current_prim, i, j, ig
      integer current_basis_idx
      integer me
      logical status

      me = ga_nodeid()

      if (.not. bas_numcont(basis, nshells))
     $     call errquit('dft_gpu_upload_basis: bas_numcont',0,
     $     BASIS_ERR)

c     First pass: Count GPU shells and primitives
c     We split SP shells (type=-1) into S and P shells.
c     We also handle general contractions by splitting them if needed,
c     but for now assume ngeneral=1 unless SP.

      nshells_gpu = 0
      nprim_total = 0

      do ishell = 1, nshells
         if (.not. bas_continfo(basis, ishell, type, nprim, ngeneral,
     $        sphcart))
     $        call errquit('dft_gpu: bas_continfo', ishell,
     $        BASIS_ERR)

         if (type .eq. -1) then
c           SP shell -> Split into S and P
            nshells_gpu = nshells_gpu + 2
            nprim_total = nprim_total + 2 * nprim
         else
c           Normal shell
c           If ngeneral > 1, we treat it as multiple shells sharing exponents?
c           Or just one shell? For Rys quadrature, usually handled as one.
c           But my C struct has one ptr_coef.
c           For Phase 1, assume ngeneral=1 for non-SP.
            if (ngeneral .gt. 1) then
               write(luout,*) 'NWDFT GPU: General contraction',
     $              ' not supported yet (shell', ishell, ')'
               call errquit('dft_gpu: general contraction',0,0)
            endif
            nshells_gpu = nshells_gpu + 1
            nprim_total = nprim_total + nprim
         endif
      end do

c     Allocate flattened arrays
      allocate(ang(nshells_gpu))
      allocate(nprim_per_shell(nshells_gpu))
      allocate(basis_idx(nshells_gpu))
      allocate(coords(3, nshells_gpu))
      
c     We allocate enough space for exponents and coefs
      allocate(exps_all(nprim_total))
      allocate(coefs_all(nprim_total))

c     Second pass: Fill arrays
      nshells_gpu = 0 ! Reset counter to use as index
      current_prim = 1
      current_basis_idx = 0 ! 0-based index for C

      do ishell = 1, nshells
         if (.not. bas_continfo(basis, ishell, type, nprim, ngeneral,
     $        sphcart)) call errquit('dft_gpu: continfo',0,BASIS_ERR)

c        Get coordinates
         if (.not. bas_cn2ce(basis, ishell, iatom))
     $        call errquit('dft_gpu: bas_cn2ce', ishell, BASIS_ERR)
         if (.not. geom_cent_get(geom, iatom, tag, 
     $        coords(1, nshells_gpu+1), charge))
     $        call errquit('dft_gpu: geom_cent_get',0,GEOM_ERR)

c        Get exponents and coefficients
         allocate(buf_exps(nprim))
         allocate(buf_coefs(nprim * ngeneral))

         if (.not. bas_get_exponent(basis, ishell, buf_exps))
     $        call errquit('dft_gpu: bas_get_exponent',0,BASIS_ERR)
         if (.not. bas_get_coeff(basis, ishell, buf_coefs))
     $        call errquit('dft_gpu: bas_get_coeff',0,BASIS_ERR)

         if (type .eq. -1) then
c           --- Split SP Shell ---
            
c           1. S-component (L=0)
            nshells_gpu = nshells_gpu + 1
            ang(nshells_gpu) = 0
            nprim_per_shell(nshells_gpu) = nprim
            basis_idx(nshells_gpu) = current_basis_idx
            current_basis_idx = current_basis_idx + 1 ! S has 1 function
            
c           Copy coords (same as retrieved above, but need to replicate for P)
            ! coords(1..3, nshells_gpu) is already set by geom_cent_get above for the first one?
            ! wait, I passed `coords(1, nshells_gpu+1)` to geom_cent_get.
            ! So coords for S is set.
            
            do i = 1, nprim
               exps_all(current_prim + i - 1) = buf_exps(i)
               coefs_all(current_prim + i - 1) = buf_coefs(i) ! 1st col is S
            end do
            current_prim = current_prim + nprim

c           2. P-component (L=1)
            nshells_gpu = nshells_gpu + 1
            ang(nshells_gpu) = 1
            nprim_per_shell(nshells_gpu) = nprim
            basis_idx(nshells_gpu) = current_basis_idx
            current_basis_idx = current_basis_idx + 3 ! P has 3 functions

            ! Copy coords from S shell
            coords(1, nshells_gpu) = coords(1, nshells_gpu-1)
            coords(2, nshells_gpu) = coords(2, nshells_gpu-1)
            coords(3, nshells_gpu) = coords(3, nshells_gpu-1)

            do i = 1, nprim
               exps_all(current_prim + i - 1) = buf_exps(i)
               coefs_all(current_prim + i - 1) = buf_coefs(nprim + i) ! 2nd col is P
            end do
            current_prim = current_prim + nprim

         else
c           --- Normal Shell ---
            nshells_gpu = nshells_gpu + 1
            ang(nshells_gpu) = type
            nprim_per_shell(nshells_gpu) = nprim
            basis_idx(nshells_gpu) = current_basis_idx
            
            if (type .eq. 0) then
                current_basis_idx = current_basis_idx + 1
            elseif (type .eq. 1) then
                current_basis_idx = current_basis_idx + 3
            elseif (type .eq. 2) then
                if (sphcart .eq. 1) then
                   current_basis_idx = current_basis_idx + 5
                else
                   current_basis_idx = current_basis_idx + 6
                endif
            elseif (type .eq. 3) then
                if (sphcart .eq. 1) then
                   current_basis_idx = current_basis_idx + 7
                else
                   current_basis_idx = current_basis_idx + 10
                endif
            elseif (type .eq. 4) then
                if (sphcart .eq. 1) then
                   current_basis_idx = current_basis_idx + 9
                else
                   current_basis_idx = current_basis_idx + 15
                endif
            else
                ! Add support for H etc. later.
                write(luout,*) 'NWDFT GPU: Shell type', type, 
     $               ' not supported yet'
                call errquit('dft_gpu: unsupported type',0,0)
            endif
            
            ! coords already set

            do i = 1, nprim
               exps_all(current_prim + i - 1) = buf_exps(i)
               coefs_all(current_prim + i - 1) = buf_coefs(i)
            end do
            current_prim = current_prim + nprim
         endif

         deallocate(buf_exps)
         deallocate(buf_coefs)

      end do

c     Debug: print basis info before upload
      if (me.eq.0) then
         write(6,*) 'FORTRAN BASIS UPLOAD DEBUG:'
         write(6,*) 'nshells_gpu=', nshells_gpu
         write(6,*) 'nprim_total=', nprim_total
         write(6,*) 'All shells:'
         j = 1
         do i = 1, nshells_gpu
            write(6,'(A,I3,A,I2,A,I3,A,I3)') '  Shell ', i, ': L=',
     $           ang(i), ' nprim=', nprim_per_shell(i),
     $           ' basis_idx=', basis_idx(i)
            j = j + nprim_per_shell(i)
         enddo
      endif

c     Call C wrapper
      call nwdft_gpu_upload_basis_c(nshells_gpu, nprim_total,
     $     coords, ang, nprim_per_shell, basis_idx, exps_all, coefs_all)

      deallocate(ang)
      deallocate(nprim_per_shell)
      deallocate(basis_idx)
      deallocate(coords)
      deallocate(exps_all)
      deallocate(coefs_all)

      end subroutine
