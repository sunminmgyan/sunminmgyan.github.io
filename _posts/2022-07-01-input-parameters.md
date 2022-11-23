---
title:  --Input parameters
author: Mingyan Sun 
date: 2022-07-01
layout: post
---

为了完成负责的等离子体计算，RGB-Maxwell拥有较高的灵活性。可供用户定义的参数较多，这一部分主要对RGB-Maxwell程序中
用户能够进行调用的参数进行介绍。

- 微分参数



```
dx, dy, dz:       Infinitesimal differences used in the spatial domain. 
dpx, dpy, dpz:    Infinitesimal difference in momentum domain
dx_o, dy_o, dz_o: Infinitesimal difference of the spatial coordinates in the observation region
dx_s, dy_s, dz_s: Infinitesimal difference of the spatial coordinates in the source region
dt :              The infinitesimal time for each time step updation.
```



- 网格参数

  

```
nx, ny, nz:        Number of spatial grids.
npx, npy, npz:     Number of momentum grids.
nx_o, ny_o, nz_o:  Number of spatial grids in the observation region.
nx_s, ny_s, nz_s:  Number of spatial grids in the source region.

```



- 边界参数



```
boundary_configuration:        Dictionary of boundary configurations. Each (key, value) pair stands for the boundary
x_left_bound_o, y_left_bound_o, z_left_bound_o:
                               the left boundaries of the spatial sub-region in the observation region.
x_left_bound_s, y_left_bound_s, z_left_bound_s:
                               the left boundaries of the spatial sub-region in the source region.
```



- 粒子参数



```
f:				 Distribution functions in different spatial regions.
particle_type:   List of particle types for each particle species.
masses:          List of masses for each particle species.
charges:         List of charges for each particle species.
degeneracy:      List of degenaracies for each particle species.
```



- 碰撞参数



```python
flavor: 		all possible collisions for the given final particle
collision_type: an index indicate which collision type the process belongs to
particle_type:  particle types correspond to different particles
```





- 在输入参数之后程序会根据输入参数的类型进行判断，如果出现输入参数的数据类型错误的情况会提供错误报表。

```python
def check_input_legacy(f, dt, \
                       nx_o, ny_o, nz_o, dx, dy, dz, boundary_configuration, \
                       x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                       npx, npy, npz, half_px, half_py, half_pz,\
                       masses, charges,\
                       sub_region_relations,\
                       num_gpus_for_each_region,\
                       num_samples,\
                       flavor, collision_type, particle_type,\
                       degeneracy, expected_collision_type):
    
    '''Check the legacy of the initial input'''
    
    if type(num_samples) != int:
        raise AssertionError("Type of num_samples should be int, but {} detected.".format(type(num_samples))) 
        
    if type(num_gpus_for_each_region) != float and type(num_gpus_for_each_region) != int:
        raise AssertionError("Type of num_gpus_for_each_region should be (float or int), but {} detected.".format(type(num_gpus_for_each_region)))
        
    if type(dt) != float and type(dt) != int:
        raise AssertionError("Type of dt should be (float or int), but {} detected.".format(type(dt))) 
        
    if type(dx) != float and type(dx) != int:
        raise AssertionError("Type of dx should be (float or int), but {} detected.".format(type(dx)))
        
    if type(dy) != float and type(dy) != int:
        raise AssertionError("Type of dy should be (float or int), but {} detected.".format(type(dy)))
        
    if type(dz) != float and type(dz) != int:
        raise AssertionError("Type of dz should be (float or int), but {} detected.".format(type(dz)))
    
    if type(f) == dict:
        for i_reg in range(len(f)):
            if type(f[i_reg]) != np.ndarray:
                raise AssertionError("Types of values in f should be numpy.ndarray, but {} detected.".format(type(f[i_reg]))) 
    else:
        raise AssertionError("Type of f should be dict, but {} detected.".format(type(f))) 
        
    if type(nx_o) == list:
        for i_reg in range(len(nx_o)):
            if type(nx_o[i_reg]) != int:
                raise AssertionError("Types of values in nx_o should be int, but {} detected.".format(type(nx_o[i_reg]))) 
    else:
        raise AssertionError("Type of nx_o should be list, but {} detected.".format(type(nx_o)))
        
    if type(ny_o) == list:
        for i_reg in range(len(ny_o)):
            if type(ny_o[i_reg]) != int:
                raise AssertionError("Types of values in ny_o should be int, but {} detected.".format(type(ny_o[i_reg]))) 
    else:
        raise AssertionError("Type of ny_o should be list, but {} detected.".format(type(ny_o)))
        
    if type(nz_o) == list:
        for i_reg in range(len(nz_o)):
            if type(nz_o[i_reg]) != int:
                raise AssertionError("Types of values in nz_o should be int, but {} detected.".format(type(nz_o[i_reg]))) 
    else:
        raise AssertionError("Type of nz_o should be list, but {} detected.".format(type(nz_o)))
        
    if type(npx) != int:
        raise AssertionError("Type of npx should be int, but {} detected.".format(type(npx))) 
         
    if type(npy) != int:
        raise AssertionError("Type of npy should be int, but {} detected.".format(type(npy)))
        
    if type(npz) != int:
        raise AssertionError("Type of npz should be int, but {} detected.".format(type(npz)))
        
    if type(boundary_configuration) == dict:
        for i_reg in range(len(boundary_configuration)):
            if type(boundary_configuration[i_reg]) == tuple:
                for i in range(len(boundary_configuration[i_reg])):
                    if type(boundary_configuration[i_reg][i]) != np.ndarray:
                        raise AssertionError("Types of boundary_configuration[#][#] should be numpy.ndarray, but {} detected.".format(type(boundary_configuration[i_reg][i]))) 
            else:
                raise AssertionError("Types of boundary_configuration[#] should be tuple, but {} detected.".format(type(boundary_configuration[i_reg]))) 
    else:
        raise AssertionError("Type of boundary_configuration should be dict, but {} detected.".format(type(boundary_configuration)))
        
    if type(x_left_bound_o) == list:
        for i_reg in range(len(x_left_bound_o)):
            if type(x_left_bound_o[i_reg]) != int and type(x_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of x_left_bound_o[#] should be (float or int), but {} detected.".format(type(x_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of x_left_bound_o should be list, but {} detected.".format(type(x_left_bound_o)))
        
    if type(y_left_bound_o) == list:
        for i_reg in range(len(y_left_bound_o)):
            if type(y_left_bound_o[i_reg]) != int and type(y_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of y_left_bound_o[#] should be (float or int), but {} detected.".format(type(y_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of y_left_bound_o should be list, but {} detected.".format(type(y_left_bound_o)))
        
    if type(z_left_bound_o) == list:
        for i_reg in range(len(z_left_bound_o)):
            if type(z_left_bound_o[i_reg]) != int and type(z_left_bound_o[i_reg]) != float:
                raise AssertionError("Types of z_left_bound_o[#] should be (float or int), but {} detected.".format(type(z_left_bound_o[i_reg]))) 
    else:
        raise AssertionError("Type of z_left_bound_o should be list, but {} detected.".format(type(z_left_bound_o)))
        
    name = ['half_px', 'half_py', 'half_pz', 'masses', 'charges']
    para = [half_px, half_py, half_pz, masses, charges]
    shape = []
    for i in range(len(name)):
        shape.append(para[i].shape)
        if type(para[i]) != np.ndarray:
            raise AssertionError("Type of {} should be numpy.ndarray, but {} detected.".format(name[i], type(para[i])))
    for i in range(len(shape)):
        for j in range(i+1, len(shape)):
            if shape[i] != shape[j]:
                raise AssertionError("Shape of {} should be same with {}, but detected {} is of shape {} and {} is of shape {}."\
                                     .format(name[i], name[j], name[i], shape[i], name[j], shape[j]))
            
    if type(sub_region_relations) == dict:
        for i_reg in ['indicator', 'position']:
            if type(sub_region_relations[i_reg]) != list:
                raise AssertionError("Types sub_region_relations[#] should be list, but {} detected.".format(type(sub_region_relations[i_reg]))) 
            else:
                for i in range(len(sub_region_relations[i_reg])):
                    if type(sub_region_relations[i_reg][i]) == list:
                        for j in range(len(sub_region_relations[i_reg][i])):
                            if type(sub_region_relations[i_reg][i][j]) != int and type(sub_region_relations[i_reg][i][j]) != type(None):
                                raise AssertionError("Types sub_region_relations[#][#][#] should be (int or None), but {} detected.".format(type(sub_region_relations[i_reg][i][j]))) 
                    else:
                        raise AssertionError("Types sub_region_relations[#][#] should be list, but {} detected.".format(type(sub_region_relations[i_reg][i]))) 
    else:
        raise AssertionError("Type of sub_region_relations should be dict, but {} detected.".format(type(sub_region_relations))) 
        
        if type(flavor) != dict:
            raise AssertionError("Type of flavor should be dict, but {} detected.".format(type(flavor)))
        if type(collision_type) != dict:
            raise AssertionError("Type of collision_type should be dict, but {} detected.".format(type(collision_type)))
            
        for key in expected_collision_type:
            if flavor[key]==None:
                raise AssertionError("flavor[{}] is None, remove {} in expected_collision_type".format(key,key))
            if collision_type[key]==None:
                raise AssertionError("collision_type[{}] is None, remove {} in expected_collision_type".format(key,key))
```

