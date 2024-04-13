**********************************************************************
      program xtob
**********************************************************************
      implicit double precision (a-h,o-z),integer(i-n)
      dimension c(50000,3),qa(50000),vhulp(3)

      character*2 qa,qr
      character*1 qcoo
      character*20 qmol
      character*60 qout,qfile
      character*100 qtemp
**********************************************************************
*                                                                    *
*     Read in xyz-format                                             *
*                                                                    *
**********************************************************************
      iperiodic=0
      izero=0
      zero=0.0
      write (6,*)'Input file in .xyz-format'
      read (5,'(a60)')qfile
      write (6,*)'Do you want to input cell parameters (y/n) ?'
      read (5,'(a1)')qcoo
      if (qcoo.eq.'y') then
      iperiodic=1
      write (6,*)'x y z dimensions periodic box in Angstrom'
      read (5,*)vx,vy,vz
      write (6,*)'a b c angles periodic box in degees'
      read (5,*)aa,ba,ca
      end if
      write (6,*)'Choose an option:'
      write (6,*)'0: do not sort'
      write (6,*)'1: sort by x-coordinate'
      write (6,*)'2: sort by y-coordinate'
      write (6,*)'3: sort by z-coordinate'
      write (6,*)'Which option (0-3) ?'
      read (5,*)ioption
      open (3,file=qfile,status='old')
    2 read (3,*,end=50,err=50)nat
      read (3,110)qmol
      do i=1,nat
      read (3,'(a100)')qtemp
      istart=3
      read(qtemp(1:2),'(a2)')qr
      if (qr(1:1).eq.' ') then
      read(qtemp(2:3),'(a2)')qr
      istart=4
      if (qr(1:1).eq.' ') then
      read(qtemp(3:4),'(a2)')qr
      istart=5
      end if
      end if
      qa(i)=qr(1:2)
*     if (qr(1:1).eq.'C ') qa(i)='C '
*     if (qr(1:2).eq.'Ca') qa(i)='Ca'
*     if (qr(1:2).eq.'Cl') qa(i)='Cl'
*     if (qr(1:1).eq.'H ') qa(i)='H '
*     if (qr(1:2).eq.'He') qa(i)='He'
*     if (qr(1:1).eq.'N ') qa(i)='N '
*     if (qr(1:2).eq.'Ni') qa(i)='Ni'
*     if (qr(1:1).eq.'O ') qa(i)='O '
*     if (qr(1:1).eq.'B ') qa(i)='B '
*     if (qr(1:1).eq.'F ') qa(i)='F '
*     if (qr(1:1).eq.'P ') qa(i)='P '
*     if (qr(1:1).eq.'S ') qa(i)='S '
*     if (qr(1:1).eq.'K ') qa(i)='K '
*     if (qr(1:1).eq.'Y ') qa(i)='Y '
*     if (qr(1:2).eq.'Al ') qa(i)='Al'
*     if (qr(1:2).eq.'Mg ') qa(i)='Mg'
*     if (qr(1:2).eq.'Si') qa(i)='Si'
*     if (qr(1:2).eq.'Se') qa(i)='Se'
*     if (qr(1:2).eq.'Rb') qa(i)='Rb'
*     if (qr(1:2).eq.'Pt') qa(i)='Pt'
*     if (qr(1:2).eq.'Ru') qa(i)='Ru'
*     if (qr(1:2).eq.'Mo') qa(i)='Mo'
*     if (qr(1:2).eq.'Ar') qa(i)='Ar'
*     if (qr(1:2).eq.'Zr') qa(i)='Zr'
*     if (qr(1:2).eq.'Ba') qa(i)='Ba'
*     if (qr(1:2).eq.'X ') qa(i)='X '
*     read(qtemp(3:100),*)c(i,1),c(i,2),c(i,3)
      istart=1
      call stranal(qtemp,istart,iend,qout,vout,iout,1)
      istart=iend
      call stranal(qtemp,istart,iend,qout,vout,iout,1)
      c(i,1)=vout
      istart=iend
      call stranal(qtemp,istart,iend,qout,vout,iout,1)
      c(i,2)=vout
      istart=iend
      call stranal(qtemp,istart,iend,qout,vout,iout,1)
      c(i,3)=vout
      end do
      if (ioption.gt.0) then
**********************************************************************
*                                                                    *
*     Sort cartesian coordinates by x- y- or z-coordinate            *
*                                                                    *
**********************************************************************
      do i1=1,nat-1
      vhulp(1)=c(i1,1)
      vhulp(2)=c(i1,2)
      vhulp(3)=c(i1,3)
      ihulp=i1
      do i2=i1+1,nat
      if (c(i2,ioption).lt.vhulp(ioption)) then
      do k1=1,3
      vhulp(k1)=c(i2,k1)
      end do
      ihulp=i2
      end if
      end do
      do k1=1,3
      qr=qa(ihulp)
      c(ihulp,k1)=c(i1,k1)
      qa(ihulp)=qa(i1)
      c(i1,k1)=vhulp(k1)
      qa(i1)=qr
      end do
      end do
      end if
**********************************************************************
*                                                                    *
*     Output .bgf-format in fort.15                                  *
*                                                                    *
**********************************************************************
    5 if (iperiodic.eq.0) write (15,500)
      if (iperiodic.eq.1) write (15,510)
      write (15,520)qmol
      write (15,530)
      if (ioption.eq.1) write (15,531)
      if (ioption.eq.2) write (15,532)
      if (ioption.eq.3) write (15,533)
      if (iperiodic.eq.1) write (15,540)vx,vy,vz,aa,ba,ca
      do 27 i2=1,nat
   27 write (15,550)i2,qa(i2),c(i2,1),c(i2,2),c(i2,3),qa(i2),izero,
     $izero,vzero
      write (15,600)
      write (15,*)
**********************************************************************
*                                                                    *
*     Output .xyz-format in fort.16                                  *
*                                                                    *
**********************************************************************
      write (16,'(i4)')nat
      write (16,'(a60)')qmol
      do i1=1,nat
      write (16,'(a2,3f15.10)')qa(i1),c(i1,1),c(i1,2),c(i1,3)
      end do

      goto 2 !next geometry
**********************************************************************
   50 continue
      close (3)
      stop 'Normal end of program; output .bgf-file(s) in fort.15'
**********************************************************************
*                                                                    *
*     Format part                                                    *
*                                                                    *
**********************************************************************
  100 format (i4)
  110 format (a20)
  120 format (a2,1x,f8.5,1x,f9.5,1x,f9.5)
  130 format (i4,1x,a2,3x,3d22.15)
  155 format ('  C',1x,a20)
  175 format (i4,1x,a2,3x,3d22.14)
  500 format ('BIOGRF 200')
  510 format ('XTLGRF 200')
  520 format ('DESCRP ',a20)
  530 format ('REMARK .bgf-file generated by xtob-script')
  531 format ('REMARK Structure sorted by x-coordinate')
  532 format ('REMARK Structure sorted by y-coordinate')
  533 format ('REMARK Structure sorted by z-coordinate')
  540 format ('CRYSTX ',6f11.5)
  550 format ('HETATM',1x,i5,1x,a2,3x,1x,3x,1x,1x,1x,5x,3f10.5,1x,
     $a5,i3,i2,1x,f8.5)
  600 format ('END')
      end
************************************************************************
************************************************************************

      subroutine stranal(qhulp,istart,iend,qout,vout,iout,icheck)

************************************************************************
      implicit double precision (a-h,o-z),integer(i-n)
      character*100 qhulp
      character*40 qout
      character*1 qchar
      dimension qchar(5)
**********************************************************************
*                                                                    *
*     Analyze string for special characters; find words in string    *
*                                                                    *
**********************************************************************
      qchar(1)=' '
      qchar(2)='/'

      ifound1=0
      do i1=istart,100
      ifound2=0
      do i2=1,icheck

      if (qhulp(i1:i1).eq.qchar(i2)) then
      ifound2=1
      if (ifound1.eq.1) then     !End of word
      iend=i1
      goto 10
      end if

      end if

      end do

      if (ifound2.eq.0.and.ifound1.eq.0) then     !Start of word
      istart2=i1
      ifound1=1
      end if

      end do

   10 continue
      qout=' '
      vout=0.0
      iout=0

      if (ifound1.eq.1) then
      qout=qhulp(istart2:iend-1)
      istart=istart2
      vout=0.0
      read (qout,*,end=20,err=20) vout
   20 iout=int(vout)
      end if

      return
      end
************************************************************************
************************************************************************
