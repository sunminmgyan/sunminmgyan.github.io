---
title: ----Vlasov_Drifit_terms
author: Mingyan Sun
date: 2022-07-08
category: Jekyll
layout: post
---

该程序主要利用差分的方式对Vlsov和Drifit项进行计算。

![Jefimenko][Jefimenko1]

##  程序的优势

  - 利用CUDA快速并行处理Vlsov和Drifit项

## 程序的结构
程序由以下函数构成：
  - Vlasov_terms（）
    - force_term_kernel()
  - Drift_terms（）
    - velocity_term_kernel()
    - left_bound_detect()  
    - right_bound_detect()

接下来将逐一对这些程序进行解析
### 程序初始化——参数输入
  - 用户需要自行定义的参数
      - 网格参数
        - nx, ny, nz 位置网格数量
        - npx, npy, npz 动量网格数量
        - total_phase_grids 总网格数量
        - half_px, half_py, half_pz 动量网格大小的一半
        - x_bound_config, y_bound_config, z_bound_config 位置边界信息
      - 粒子参数
        - f_x_p_t 粒子分布函数；
        - Fx, Fy, Fz 外力分布；
        - force_term Drifit初始化
        - masses 粒子质量；
      - 差分，GPU参数
        - num_of_particle_types 粒子类型数目；
        - number_momentum_levels 向量分层数目；
        - dt, dx, dy, dz 时间与位置差分；
        - dpx, dpy, dpz 动量差分；
        - threadsperblock，blockspergrid_total_phase GPU线程，块数目；
        
### Vlasov_terms()—— Vlasov项计算

该函数主要完成对Vlasov项的计算：

![Vlasov][Vlasov1]

- 首先进行结果的初始化，并调用kernel函数进行并行化处理:
```python
def Vlasov_terms(f_x_p_t, Fx, Fy, Fz, \
                 masses, total_grid, num_of_particle_types, \
                 npx, npy, npz, nx, ny, nz, \
                 half_px, half_py, half_pz, \
                 dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
                 x_bound_config, y_bound_config, z_bound_config, \
                 blockspergrid_total_phase, threadsperblock):
    force_term = cupy.zeros([number_momentum_levels,num_of_particle_types,nx*ny*nz*npx*npy*npz])
    
    # Vlasov term
    force_term_kernel[blockspergrid_total_phase, threadsperblock]\
    (f_x_p_t, force_term, Fx, Fy, Fz, \
    total_grid, num_of_particle_types, \
    npx, npy, npz, nx, ny, nz, \
    dpx, dpy, dpz, number_momentum_levels)
    
    return force_term
```
- 其次进行GPU线程与真实向量，位置索引的对应:
```python
i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ipz = i_grid%npz
        ipz_rest = i_grid//npz
        ipy = ipz_rest%npy
        ipy_rest = ipz_rest//npy
        ipx = ipy_rest%npx
        ipx_rest = ipy_rest//npx
        iz = ipx_rest%nz
        iz_rest = ipx_rest//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
    
        # enforce periodical boundary conditions, ipxPlus --> ipx+1
        ipxPlus = (ipx+1)%npx
        ipyPlus = (ipy+1)%npy
        ipzPlus = (ipz+1)%npz
    
        # -1%3 should be 2, but cuda yields 0, so we use
        ipxMinus = (ipx+npx-1)%npx
        ipyMinus = (ipy+npy-1)%npy
        ipzMinus = (ipz+npz-1)%npz
        
        # convert six-d to one-d 
        i_phasexmin = threeD_to_oneD(ix, iy, iz, ipxMinus, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phasexmax = threeD_to_oneD(ix, iy, iz, ipxPlus, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phaseymin = threeD_to_oneD(ix, iy, iz, ipx, ipyMinus, ipz, nx, ny, nz, npx, npy, npz)
        i_phaseymax = threeD_to_oneD(ix, iy, iz, ipx, ipyPlus, ipz, nx, ny, nz, npx, npy, npz)
        i_phasezmin = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzMinus, nx, ny, nz, npx, npy, npz)
        i_phasezmax = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzPlus, nx, ny, nz, npx, npy, npz)
```

- 最终进行Vlasov项的计算:

```python
def pf_pp(Fx, dx, fleftx, fcurrent, frightx):
    if Fx > 0:
        return (fcurrent - fleftx)/dx
    return (frightx - fcurrent)/dx
```

```python
for p_type in range(num_of_particle_types):

    # loop through all momentum levels
    for i_level in range(number_momentum_levels):  
        
        # distribution functions at p, p-dpx, px+dpx
        fcurrentp = f_x_p_t[i_level, p_type, i_grid]
        fleftpx = f_x_p_t[i_level, p_type, i_phasexmin]
        frightpx = f_x_p_t[i_level, p_type, i_phasexmax]
        fleftpy = f_x_p_t[i_level, p_type, i_phaseymin]
        frightpy = f_x_p_t[i_level, p_type, i_phaseymax]
        fleftpz = f_x_p_t[i_level, p_type, i_phasezmin]
        frightpz = f_x_p_t[i_level, p_type, i_phasezmax]
    
        # External forces at p, p-dpx, px+dpx
        Fcurrentpx = Fx[i_level, p_type, i_grid]
        Fcurrentpy = Fy[i_level, p_type, i_grid]
        Fcurrentpz = Fz[i_level, p_type, i_grid]
    
        force_term[i_level, p_type, i_grid] = Fcurrentpx*pf_pp(Fcurrentpx, dpx[p_type]/(npx**i_level), fleftpx, fcurrentp, frightpx) + \
                                              Fcurrentpy*pf_pp(Fcurrentpy, dpy[p_type]/(npy**i_level), fleftpy, fcurrentp, frightpy) + \
                                              Fcurrentpz*pf_pp(Fcurrentpz, dpz[p_type]/(npz**i_level), fleftpz, fcurrentp, frightpz)
```

### Drift_terms()—— Drift项计算

该函数主要完成对Drift项的计算：

![Drifit][Drifit1]

- 进行结果的初始化，并调用kernel函数进行并行化处理:
```python
def Drift_terms(f_x_p_t, masses, total_grid, num_of_particle_types, \
                npx, npy, npz, nx, ny, nz, \
                half_px, half_py, half_pz, \
                dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
                x_bound_config, y_bound_config, z_bound_config, \
                blockspergrid_total_phase, threadsperblock):

    velocity_term = cupy.zeros([number_momentum_levels,num_of_particle_types,nx*ny*nz*npx*npy*npz])
    
    # drift term
    velocity_term_kernel[blockspergrid_total_phase, threadsperblock]\
    (f_x_p_t, velocity_term, masses, \
     total_grid, num_of_particle_types, \
     npx, npy, npz, nx, ny, nz, \
     half_px, half_py, half_pz, \
     dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
     x_bound_config, y_bound_config, z_bound_config)
    
    return velocity_term
```
- 进行GPU线程与真实向量，位置索引的对应:

```python
    i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ipz = i_grid%npz
        ipz_rest = i_grid//npz
        ipy = ipz_rest%npy
        ipy_rest = ipz_rest//npy
        ipx = ipy_rest%npx
        ipx_rest = ipy_rest//npx
        iz = ipx_rest%nz
        iz_rest = ipx_rest//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
        
        # distribution functions out of the computation domain of each card are always set to be 0
        ixPlus = (ix+1)
        iyPlus = (iy+1)
        izPlus = (iz+1)
        ixMinus = (ix-1)
        iyMinus = (iy-1)
        izMinus = (iz-1)

        # convert six-d to one-d 
        i_phasexmin = threeD_to_oneD(ixMinus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phasexmax = threeD_to_oneD(ixPlus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phaseymin = threeD_to_oneD(ix, iyMinus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phaseymax = threeD_to_oneD(ix, iyPlus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phasezmin = threeD_to_oneD(ix, iy, izMinus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
        i_phasezmax = threeD_to_oneD(ix, iy, izPlus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
```

- 进行px,py,pz,Ep的计算:

```python
for p_type in range(num_of_particle_types):    
            
    # masses
    mp_squared = masses[p_type]**2
    
    # loop through all momentum levels
    for i_level in range(number_momentum_levels):
        
        # acquire p from the central value
        # Note that for different particles, they have different dpx and px_left_bound
        # the momentum level corresponds to the level of straitification
        px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])/(npx**i_level)
        py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])/(npy**i_level)
        pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])/(npz**i_level)

        # energy for current grid
        Ep = math.sqrt(mp_squared+px**2+py**2+pz**2)
```

- 计算vx,vy,vz,同时进行差分计算的准备工作：

```python
vx = 0.
vy = 0.
vz = 0.
if Ep > 10**-10:
    vx = px/Ep
    vy = py/Ep
    vz = pz/Ep

# distribution functions at x-dx, x, x+dx
fcurrent = f_x_p_t[i_level, p_type, i_grid]                
fleftx = left_bound_detect(ixMinus, f_x_p_t, i_level, p_type, i_phasexmin)
flefty = left_bound_detect(iyMinus, f_x_p_t, i_level, p_type, i_phaseymin)
fleftz = left_bound_detect(izMinus, f_x_p_t, i_level, p_type, i_phasezmin)
frightx = right_bound_detect(ixPlus, nx, f_x_p_t, i_level, p_type, i_phasexmax)
frighty = right_bound_detect(iyPlus, ny, f_x_p_t, i_level, p_type, i_phaseymax)
frightz = right_bound_detect(izPlus, nz, f_x_p_t, i_level, p_type, i_phasezmax)
```
- 计算Vlasov项，并利用cuda原子操作进行求和：

```python
if vx > 0:
    gradx = vx*(fcurrent - fleftx)/dx
    # if vx > 0, the right most boundary needs to be considered
    if ix > (nx - 1.5):
        # mirror index for ipx
        ipx_mirror = mirror(npx, ipx)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx_mirror, ipy, ipz, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), -vx*(fcurrent)/dx*(1-x_bound_config[iy, iz]))
else:
    gradx = vx*(frightx - fcurrent)/dx
    # if vx < 0., the left most boundary needs to be considered                    
    if ix < 0.5:
        # mirror in dex for ipx
        ipx_mirror = mirror(npx, ipx)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx_mirror, ipy, ipz, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), vx*(fcurrent)/dx*(1-x_bound_config[iy, iz]))
##########################################################################################################
if vy > 0:
    grady = vy*(fcurrent - flefty)/dy
    # if vy > 0, the right most boundary needs to be considered
    if iy > (ny - 1.5):
        # mirror index for ipy
        ipy_mirror = mirror(npy, ipy)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy_mirror, ipz, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), -vy*(fcurrent)/dy*(1-y_bound_config[iz, ix]))
else:
    grady = vy*(frighty - fcurrent)/dy
    # if vy < 0., the left most boundary needs to be considered                   
    if iy < 0.5:
        # mirror index for ipy
        ipy_mirror = mirror(npy, ipy)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy_mirror, ipz, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), vy*(fcurrent)/dy*(1-y_bound_config[iz, ix]))
###########################################################################################
if vz > 0:
    gradz = vz*(fcurrent - fleftz)/dz
    # if vz > 0, the right most boundary needs to be considered
    if iz > (nz - 1.5):
        # mirror index for ipz
        ipz_mirror = mirror(npz, ipz)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz_mirror, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), -vz*(fcurrent)/dz*(1-z_bound_config[ix, iy]))
else:
    gradz = vz*(frightz - fcurrent)/dz
    # if vz < 0., the left most boundary needs to be considered
    if iz < 0.5:
        # mirror index for ipz
        ipz_mirror = mirror(npz, ipz)
        i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz_mirror, nx, ny, nz, npx, npy, npz)
        
        # store the values for the boundaries at mirror point
        cuda.atomic.add(velocity_term, (i_level, p_type, i_grid_mirror), vz*(fcurrent)/dz*(1-z_bound_config[ix, iy]))

# the value-updation at i_grid should be taken into account no matter 
# its reflection or absorption. If it is totally reflection, the value should 
# be stored in the mirror grid as above.
cuda.atomic.add(velocity_term, (i_level, p_type, i_grid), gradx+grady+gradz)
```



[Jefimenko1]:data:image/png;base64,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
[Drifit1]:data:image/png;base64,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
[Vlasov1]:data:image/png;base64,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
