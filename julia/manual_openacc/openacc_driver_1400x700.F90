module openacc_driver
USE, INTRINSIC :: ISO_C_BINDING
USE OPENACC

INTEGER (C_INT64_T ), PARAMETER :: NX = 1400
INTEGER (C_INT64_T ), PARAMETER :: NZ = 700
REAL (C_DOUBLE ), PARAMETER :: DX = 14.285714285714286
REAL (C_DOUBLE ), PARAMETER :: DZ = 14.285714285714286
INTEGER (C_INT64_T ), PARAMETER :: HS = 2
INTEGER (C_INT64_T ), PARAMETER :: NUM_VARS = 4
REAL (C_DOUBLE ), PARAMETER :: C0 = 27.562941092972594
REAL (C_DOUBLE ), PARAMETER :: GAMMA = 1.400278940027894
REAL (C_DOUBLE ), PARAMETER :: P0 = 100000.0
REAL (C_DOUBLE ), PARAMETER :: HV_BETA = 0.05
REAL (C_DOUBLE ), PARAMETER :: GRAV = 9.8
REAL (C_DOUBLE ), PARAMETER :: RD = 287.0
REAL (C_DOUBLE ), PARAMETER :: CP = 1004.0
REAL (C_DOUBLE ), PARAMETER :: CV = 717.0
INTEGER (C_INT64_T ), PARAMETER :: ID_DENS = 1
INTEGER (C_INT64_T ), PARAMETER :: ID_UMOM = 2
INTEGER (C_INT64_T ), PARAMETER :: ID_WMOM = 3
INTEGER (C_INT64_T ), PARAMETER :: ID_RHOT = 4
INTEGER (C_INT64_T ), PARAMETER :: STEN_SIZE = 4
INTEGER (C_INT64_T ), PARAMETER :: DATA_SPEC = 3
REAL (C_DOUBLE ), PARAMETER :: PI = 3.141592653589793
INTEGER (C_INT64_T ), PARAMETER :: I_BEG = 1
INTEGER (C_INT64_T ), PARAMETER :: K_BEG = 1
REAL (C_DOUBLE ), PARAMETER :: XLEN = 20000.0
REAL (C_DOUBLE ), PARAMETER :: ZLEN = 10000.0
INTEGER (C_INT64_T ), PARAMETER :: DATA_SPEC_GRAVITY_WAVES = 3

public jai_allocate, jai_updateto, jai_tend_x, jai_tend_z, jai_reductions
public jai_tend_apply, jai_halo_z, jai_halo_1rank, jai_halo_inject
public jai_halo_sendbuf, jai_halo_recvbuf, jai_deallocate
public jai_updatefrombuf, jai_updatetobuf
public jai_get_num_devices, jai_get_device_num, jai_set_device_num, jai_wait

contains

INTEGER (C_INT64_T) FUNCTION jai_halo_recvbuf(recvbuf_l,recvbuf_r,state) BIND(C, name="jai_halo_recvbuf")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_r
REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(OUT) :: state

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: s,k,ll
    !$acc parallel loop collapse(3) present(state, recvbuf_l, recvbuf_r) ! async
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do s = 1 , hs
          state(-hs+s,k,ll) = recvbuf_l(s,k,ll)
          state(nx+s ,k,ll) = recvbuf_r(s,k,ll)
        enddo
      enddo
    enddo

jai_halo_recvbuf = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_halo_sendbuf(state,sendbuf_l,sendbuf_r) BIND(C, name="jai_halo_sendbuf")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(OUT) :: sendbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(OUT) :: sendbuf_r

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: s,k,ll

    !$acc parallel loop collapse(3) present(state, sendbuf_l, sendbuf_r) ! async
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do s = 1 , hs
          sendbuf_l(s,k,ll) = state(s      ,k,ll)
          sendbuf_r(s,k,ll) = state(nx-hs+s,k,ll)
        enddo
      enddo
    enddo

jai_halo_sendbuf = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_num_devices(buf) BIND(C, name="jai_get_num_devices")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = acc_get_num_devices(acc_get_device_type())

jai_get_num_devices = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_device_num(buf) BIND(C, name="jai_get_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = acc_get_device_num(acc_get_device_type())

jai_get_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_set_device_num(buf) BIND(C, name="jai_set_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(IN) :: buf
INTEGER :: device_number
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

device_number = buf(1)
CALL acc_set_device_num(device_number, acc_get_device_type())

jai_set_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_wait() BIND(C, name="jai_wait")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!CALL acc_wait(INTEGER(acc_get_default_async()))
CALL acc_wait(0)

jai_wait = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_allocate(state,statetmp,flux,tend,hy_dens_cell,hy_dens_theta_cell,hy_dens_int,hy_dens_theta_int,hy_pressure_int, sendbuf_l, sendbuf_r, recvbuf_l, recvbuf_r) BIND(C, name="jai_allocate")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: statetmp
REAL (C_DOUBLE ), DIMENSION(1:1401, 1:701, 1:4), INTENT(IN) :: flux
REAL (C_DOUBLE ), DIMENSION(1:1400, 1:700, 1:4), INTENT(IN) :: tend
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_theta_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_pressure_int
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_r
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_r

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!$acc enter data create(state)

!$acc enter data create(statetmp)

!$acc enter data create(flux)

!$acc enter data create(tend)

!$acc enter data create(hy_dens_cell)

!$acc enter data create(hy_dens_theta_cell)

!$acc enter data create(hy_dens_int)

!$acc enter data create(hy_dens_theta_int)

!$acc enter data create(hy_pressure_int)

!$acc enter data create(sendbuf_l) 

!$acc enter data create(sendbuf_r) 

!$acc enter data create(recvbuf_l) 

!$acc enter data create(recvbuf_r) 

jai_allocate = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_updateto(state,statetmp,hy_dens_cell,hy_dens_theta_cell,hy_dens_int,hy_dens_theta_int,hy_pressure_int) BIND(C, name="jai_updateto")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: statetmp
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_theta_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_pressure_int

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!$acc update device(state) 

!$acc update device(statetmp) 

!$acc update device(hy_dens_cell) 

!$acc update device(hy_dens_theta_cell) 

!$acc update device(hy_dens_int) 

!$acc update device(hy_dens_theta_int) 

!$acc update device(hy_pressure_int) 

jai_updateto = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_reductions(state,hy_dens_cell,hy_dens_theta_cell,glob) BIND(C, name="jai_reductions")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell
REAL (C_DOUBLE ), DIMENSION(1:2), INTENT(OUT) :: glob

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: i, k, ierr
    real(C_DOUBLE) :: r,u,w,th,p,t,ke,ie, mass, te
    mass = 0.
    te = 0.

    !$acc parallel loop collapse(2) reduction(+:mass,te) present(state, hy_dens_cell, hy_dens_theta_cell)
    do k = 1 , nz
      do i = 1 , nx
        r  =   state(i,k,ID_DENS) + hy_dens_cell(k)             ! Density
        u  =   state(i,k,ID_UMOM) / r                           ! U-wind
        w  =   state(i,k,ID_WMOM) / r                           ! W-wind
        th = ( state(i,k,ID_RHOT) + hy_dens_theta_cell(k) ) / r ! Potential Temperature (theta)
        p  = C0*(r*th)**gamma      ! Pressure
        t  = th / (p0/p)**(rd/cp)  ! Temperature
        ke = r*(u*u+w*w)           ! Kinetic Energy
        ie = r*cv*t                ! Internal Energy
        mass = mass + r            *dx*dz ! Accumulate domain mass
        te   = te   + (ke + r*cv*t)*dx*dz ! Accumulate domain total energy
      enddo
    enddo


    glob(1) = mass
    glob(2) = te

jai_reductions = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_tend_x(state,dt,hy_dens_cell,hy_dens_theta_cell,flux,tend) BIND(C, name="jai_tend_x")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), INTENT(IN) :: dt
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell
REAL (C_DOUBLE ), DIMENSION(1:1401, 1:701, 1:4), INTENT(OUT) :: flux
REAL (C_DOUBLE ), DIMENSION(1:1400, 1:700, 1:4), INTENT(OUT) :: tend

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: i,k,ll,s
    real(C_DOUBLE) :: r,u,w,t,p, stencil(4), d3_vals(NUM_VARS), vals(NUM_VARS), hv_coef
    !Compute the hyperviscosity coeficient
    hv_coef = -hv_beta * dx / (16*dt)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Compute fluxes in the x-direction for each cell
    !$acc parallel loop collapse(2) private(stencil,vals,d3_vals) &
    !$acc& present(state, hy_dens_cell, hy_dens_theta_cell, flux) async
    do k = 1 , nz
      do i = 1 , nx+1
        !Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        do ll = 1 , NUM_VARS
          do s = 1 , sten_size
            stencil(s) = state(i-hs-1+s,k,ll)
          enddo
          !Fourth-order-accurate interpolation of the state
          vals(ll) = -stencil(1)/12 + 7*stencil(2)/12 + 7*stencil(3)/12 - stencil(4)/12
          !First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(ll) = -stencil(1) + 3*stencil(2) - 3*stencil(3) + stencil(4)
        enddo
        !Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        r = vals(ID_DENS) + hy_dens_cell(k)
        u = vals(ID_UMOM) / r
        w = vals(ID_WMOM) / r
        t = ( vals(ID_RHOT) + hy_dens_theta_cell(k) ) / r
        p = C0*(r*t)**gamma
        !Compute the flux vector
        flux(i,k,ID_DENS) = r*u     - hv_coef*d3_vals(ID_DENS)
        flux(i,k,ID_UMOM) = r*u*u+p - hv_coef*d3_vals(ID_UMOM)
        flux(i,k,ID_WMOM) = r*u*w   - hv_coef*d3_vals(ID_WMOM)
        flux(i,k,ID_RHOT) = r*u*t   - hv_coef*d3_vals(ID_RHOT)
      enddo
    enddo
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Use the fluxes to compute tendencies for each cell
    !$acc parallel loop collapse(3) present(flux, tend) async
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          tend(i,k,ll) = -( flux(i+1,k,ll) - flux(i,k,ll) ) / dx
        enddo
      enddo
    enddo

jai_tend_x = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_tend_apply(state_init,tend,hy_dens_cell,dt,state_out) BIND(C, name="jai_tend_apply")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state_init
REAL (C_DOUBLE ), DIMENSION(1:1400, 1:700, 1:4), INTENT(INOUT) :: tend
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), INTENT(IN) :: dt
REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(OUT) :: state_out

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: i,k,ll,s
    real(C_DOUBLE) :: x, z, wpert, dist, x0, z0, xrad, zrad, amp

    !Apply the tendencies to the fluid state
    !$acc parallel loop collapse(3) present(state_init, state_out, tend, hy_dens_cell) async
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          if (data_spec == DATA_SPEC_GRAVITY_WAVES) then
            x = (i_beg-1 + i-0.5_8) * dx
            z = (k_beg-1 + k-0.5_8) * dz
            ! The following requires "acc routine" in OpenACC and "declare target" in OpenMP offload
            ! Neither of these are particularly well supported by compilers, so I'm manually inlining
            ! wpert = sample_ellipse_cosine( x,z , 0.01_8 , xlen/8,1000._8, 500._8,500._8)
            x0 = xlen/8
            z0 = 1000
            xrad = 500
            zrad = 500
            amp = 0.01_8
            !Compute distance from bubble center
            dist = sqrt( ((x-x0)/xrad)**2 + ((z-z0)/zrad)**2 ) * pi / 2._8
            !If the distance from bubble center is less than the radius, create a cos**2 profile
            if (dist <= pi / 2._8) then
              wpert = amp * cos(dist)**2
            else
              wpert = 0._8
            endif
            tend(i,k,ID_WMOM) = tend(i,k,ID_WMOM) + wpert*hy_dens_cell(k)
          endif
          state_out(i,k,ll) = state_init(i,k,ll) + dt * tend(i,k,ll)
        enddo
      enddo
    enddo

jai_tend_apply = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_halo_z(state,hy_dens_cell) BIND(C, name="jai_halo_z")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(INOUT) :: state
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: i, ll

    !$acc parallel loop collapse(2) present(state, hy_dens_cell) async
    do ll = 1 , NUM_VARS
      do i = 1-hs,nx+hs
        if (ll == ID_WMOM) then
          state(i,-1  ,ll) = 0
          state(i,0   ,ll) = 0
          state(i,nz+1,ll) = 0
          state(i,nz+2,ll) = 0
        else if (ll == ID_UMOM) then
          state(i,-1  ,ll) = state(i,1 ,ll) / hy_dens_cell( 1) * hy_dens_cell(-1  )
          state(i,0   ,ll) = state(i,1 ,ll) / hy_dens_cell( 1) * hy_dens_cell( 0  )
          state(i,nz+1,ll) = state(i,nz,ll) / hy_dens_cell(nz) * hy_dens_cell(nz+1)
          state(i,nz+2,ll) = state(i,nz,ll) / hy_dens_cell(nz) * hy_dens_cell(nz+2)
        else
          state(i,-1  ,ll) = state(i,1 ,ll)
          state(i,0   ,ll) = state(i,1 ,ll)
          state(i,nz+1,ll) = state(i,nz,ll)
          state(i,nz+2,ll) = state(i,nz,ll)
        endif
      enddo
    enddo

jai_halo_z = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_tend_z(state,dt,hy_dens_int,hy_dens_theta_int,hy_pressure_int,flux,tend) BIND(C, name="jai_tend_z")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), INTENT(IN) :: dt
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_theta_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_pressure_int
REAL (C_DOUBLE ), DIMENSION(1:1401, 1:701, 1:4), INTENT(OUT) :: flux
REAL (C_DOUBLE ), DIMENSION(1:1400, 1:700, 1:4), INTENT(OUT) :: tend

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: i,k,ll,s
    real(C_DOUBLE) :: r,u,w,t,p, stencil(4), d3_vals(NUM_VARS), vals(NUM_VARS), hv_coef
    !Compute the hyperviscosity coeficient
    hv_coef = -hv_beta * dz / (16*dt)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Compute fluxes in the x-direction for each cell
    !$acc parallel loop collapse(2) private(stencil,vals,d3_vals) &
    !$acc& present(state, hy_dens_int, hy_dens_theta_int, hy_pressure_int, flux) async
    do k = 1 , nz+1
      do i = 1 , nx
        !Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        do ll = 1 , NUM_VARS
          do s = 1 , sten_size
            stencil(s) = state(i,k-hs-1+s,ll)
          enddo
          !Fourth-order-accurate interpolation of the state
          vals(ll) = -stencil(1)/12 + 7*stencil(2)/12 + 7*stencil(3)/12 - stencil(4)/12
          !First-order-accurate interpolation of the third spatial derivative of the state
          d3_vals(ll) = -stencil(1) + 3*stencil(2) - 3*stencil(3) + stencil(4)
        enddo
        !Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        r = vals(ID_DENS) + hy_dens_int(k)
        u = vals(ID_UMOM) / r
        w = vals(ID_WMOM) / r
        t = ( vals(ID_RHOT) + hy_dens_theta_int(k) ) / r
        p = C0*(r*t)**gamma - hy_pressure_int(k)
        !Enforce vertical boundary condition and exact mass conservation
        if (k == 1 .or. k == nz+1) then
          w                = 0
          d3_vals(ID_DENS) = 0
        endif
        !Compute the flux vector with hyperviscosity
        flux(i,k,ID_DENS) = r*w     - hv_coef*d3_vals(ID_DENS)
        flux(i,k,ID_UMOM) = r*w*u   - hv_coef*d3_vals(ID_UMOM)
        flux(i,k,ID_WMOM) = r*w*w+p - hv_coef*d3_vals(ID_WMOM)
        flux(i,k,ID_RHOT) = r*w*t   - hv_coef*d3_vals(ID_RHOT)
      enddo
    enddo
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Use the fluxes to compute tendencies for each cell
    !$acc parallel loop collapse(3) present(flux, state, tend) async
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          tend(i,k,ll) = -( flux(i,k+1,ll) - flux(i,k,ll) ) / dz
          if (ll == ID_WMOM) then
            tend(i,k,ID_WMOM) = tend(i,k,ID_WMOM) - state(i,k,ID_DENS)*grav
          endif
        enddo
      enddo
    enddo

jai_tend_z = JAI_ERRORCODE

END FUNCTION


INTEGER (C_INT64_T) FUNCTION jai_halo_1rank(state) BIND(C, name="jai_halo_1rank")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(INOUT) :: state

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: k,ll
    !$acc parallel loop collapse(2) present(state) async
    do ll = 1 , NUM_VARS
        do k = 1 , nz
          state(-1  ,k,ll) = state(nx-1,k,ll)
          state(0   ,k,ll) = state(nx  ,k,ll)
          state(nx+1,k,ll) = state(1   ,k,ll)
          state(nx+2,k,ll) = state(2   ,k,ll)
        enddo
    enddo

jai_halo_1rank = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_halo_inject(state, hy_dens_cell, hy_dens_theta_cell) BIND(C, name="jai_halo_inject")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(INOUT) :: state
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

    integer :: k
    real(C_DOUBLE) :: z

    !$acc parallel loop present(state, hy_dens_cell, hy_dens_theta_cell) ! async
    do k = 1 , nz
      z = (k_beg-1 + k-0.5_8)*dz
      if (abs(z-3*zlen/4) <= zlen/16) then
        state(-1:0,k,ID_UMOM) = (state(-1:0,k,ID_DENS)+hy_dens_cell(k)) * 50._8
        state(-1:0,k,ID_RHOT) = (state(-1:0,k,ID_DENS)+hy_dens_cell(k)) * 298._8 - hy_dens_theta_cell(k)
      endif
    enddo

jai_halo_inject = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_updatefrombuf(sendbuf_l, sendbuf_r) BIND(C, name="jai_updatefrombuf")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_r

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!$acc update host(sendbuf_l)

!$acc update host(sendbuf_r)

jai_updatefrombuf = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_updatetobuf(recvbuf_l,recvbuf_r) BIND(C, name="jai_updatetobuf")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_r

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!$acc update device(recvbuf_l) async

!$acc update device(recvbuf_r) async

jai_updatetobuf = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_deallocate(state,statetmp,flux,tend,hy_dens_cell,hy_dens_theta_cell,hy_dens_int,hy_dens_theta_int,hy_pressure_int,sendbuf_l,sendbuf_r,recvbuf_l,recvbuf_r) BIND(C, name="jai_deallocate")
USE, INTRINSIC :: ISO_C_BINDING

REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: state
REAL (C_DOUBLE ), DIMENSION(-1:1402, -1:702, 1:4), INTENT(IN) :: statetmp
REAL (C_DOUBLE ), DIMENSION(1:1401, 1:701, 1:4), INTENT(IN) :: flux
REAL (C_DOUBLE ), DIMENSION(1:1400, 1:700, 1:4), INTENT(IN) :: tend
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_cell
REAL (C_DOUBLE ), DIMENSION(-1:702), INTENT(IN) :: hy_dens_theta_cell
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_dens_theta_int
REAL (C_DOUBLE ), DIMENSION(1:701), INTENT(IN) :: hy_pressure_int
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: sendbuf_r
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_l
REAL (C_DOUBLE ), DIMENSION(1:2, 1:700, 1:4), INTENT(IN) :: recvbuf_r

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!$acc exit data delete(state)

!$acc exit data delete(statetmp)

!$acc exit data delete(flux)

!$acc exit data delete(tend)

!$acc exit data delete(hy_dens_cell)

!$acc exit data delete(hy_dens_theta_cell)

!$acc exit data delete(hy_dens_int)

!$acc exit data delete(hy_dens_theta_int)

!$acc exit data delete(hy_pressure_int)

!$acc exit data delete(sendbuf_l) 

!$acc exit data delete(sendbuf_r) 

!$acc exit data delete(recvbuf_l) 

!$acc exit data delete(recvbuf_r) 

jai_deallocate = JAI_ERRORCODE

END FUNCTION

end module


