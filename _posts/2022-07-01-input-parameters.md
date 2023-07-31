---
title:  Input parameters
author: Mingyan Sun 
date: 2022-07-01
layout: post
---

​	The RBG-Maxwell framework exhibits a significant level of adaptability and offers a vast array of user-defined parameters, rendering it proficient in facilitating a wide range of plasma simulations.

​	This section presents an introduction to the parameters employed in the RGB-Maxwell program.



- Differential parameters



```
dx, dy, dz:       
		Infinitesimal differences used in the spatial domain. 
		e.g. dx = dy = dz = 10**(-5)
dpx, dpy, dpz:    
		Infinitesimal difference in momentum domain
	    e.g. dpx = dpy = dpz = 10**(-25)
dx_o, dy_o, dz_o: 
		Infinitesimal difference of the spatial coordinates in the observation region
		e.g. dx_o, dy_o, dz_o = [dx]*number_regions, [dy]*number_regions, [dz]*number_regions
dt :              
		The infinitesimal time for each time step updation.
	    e.g. dt = 10**(-13)
n_step:
		number of total time steps
		e.g. n_step = 10001
drift_order:
        order of drift terms in upwind scheme
		e.g. drift_order = 2
num_samples:
		number samples gives the number of sample points in MC integration
		e.g. num_samples = 100
```



- Grid parameters

  

```
nx, ny, nz:        
		Number of spatial grids.
		e.g. nx = ny = nz = 101
npx, npy, npz:     
		Number of momentum grids.
		e.g. npx, npy, npz = 3, 4, 5
nx_o, ny_o, nz_o:  
		Number of spatial grids in the observation region.
		e.g. nx_o, ny_o, nz_o = [nx], [ny], [nz]

```



- Boundary parameters



```
x_left_bound_o, y_left_bound_o, z_left_bound_o:
        the left boundaries of the spatial sub-region in the observation region.
        e.g. x_left_bound_o, y_left_bound_o, z_left_bound_o = \
			((np.array([19, 19, 19, 19, 0, 0, 0, 0])*12/40-6)/0.197).tolist(),\
			((np.array([0, 19, 0, 19, 0, 19, 0, 19])*12/40-6)/0.197).tolist(),\
			((np.array([19, 19, 0, 0, 19, 19, 0, 0])*12/40-6)/0.197).tolist()
half_px, half_py, half_pz:
            Three lists of the momentum length in x,y and z directions.
            e.g. half_px, half_py, half_pz = np.array([-px_left_bound]*7), \
				 np.array([-py_left_bound]*7), np.array([-pz_left_bound]*7) 
x_bound_config, y_bound_config, z_bound_config:
            configuretions of the boundary conditions. 
            x_bound_config is of shape [ny, nz, 2]
            y_bound_config is of shape [nx, nz, 2]
            z_bound_config is of shape [nx, ny, 2]				 
```



- Particle parameters



```
f:				 
		Distribution functions in different spatial regions.
		e.g. f = [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz].
particle_type:  
		List of particle types for each particle species.
		e.g. particle_type = [1,1,1,1,1,1,2]
masses:          
		List of masses for each particle species.
		e.g. masses = np.array([0.3,0.3,0.5,0.3,0.3,0.5,0.5],dtype=np.float64)
charges:         
		List of charges for each particle species.
		e.g. charges = np.array([unit_charge*2/3,-unit_charge/3,-unit_charge/3,\
                    -unit_charge*2/3,unit_charge/3,unit_charge/3, 0.],dtype=np.float64)
degeneracy:      
		List of degenaracies for each particle species.
		e.g. degeneracy = np.array([6.,6.,6.,6.,6.,6.,16.])
```



- Region parameters



```python
num_gpus_for_each_region:
    	each spatial should use the full GPU, this number can be fractional if many regions are chosen and only one GPU is available
        e.g. num_gpus_for_each_region = 1.
region_id:
		the index of the current spatial region,
        e.g. region_id = 1
number_regions: 
        total number of spatial regions
        e.g. number_regions = 8
sub_region_relations:
        Dictionary of the relative locations amongest the sub-regions.
        key: 'indicator' gives the index of surfaces to be exchanged.
        key: 'position' gives the relative positions between the regions.
        e.g. sub_region_relations = \
        {'indicator': [[0,3,4],[0,2,4],[0,3,5],[0,2,5],\
                       [1,3,4],[1,2,4],[1,3,5],[1,2,5]],\
         'position': [[0,    1,    2,    3,    4,    5,    6,    7],\     -----base
                      [4,    5,    6,    7,    None, None, None, None],\  -----minus x
                      [None, None, None, None, 0,    1,    2,    3],\     -----plus x
                      [None, 0,    None, 2,    None, 4,    None, 6],\     -----minus y
                      [1,    None, 3,    None, 5,    None, 7,    None],\  -----plus y
                      [2,    3,    None, None, 6,    7,    None, None],\  -----minus z
                      [None, None, 0,    1,    None, None, 4,    5]]}     -----plus z
    	
```



- Collision parameters



```python
flavor: 		
		all possible collisions for the given final particle
        flavor = {'2TO2:, '2TO3':, ;3TO2':}
        e.g. flavor['2TO2']=np.array([[[1,0,1,0],    [10001,10001,10001,10001]],
                                     [[0,1,0,1],    [4,1,4,1]],
                                     [[0,1,3,2],    [10001,10001,10001,10001]],
                                     [[0,1,2,3],    [10001,10001,10001,10001]],
                                     [[1,4,1,4],    [10001,10001,10001,10001]]],dtype=np.int64)
collision_type: 
		an index indicate which collision type the process belongs to
        collision_type = {'2TO2:, '2TO3':, ;3TO2':}
        e.g. collision_type['2TO3']=np.array([[0,1,10001,10001],\
                                             [1,10001,10001,10001],\
                                             [2,10001,10001,10001],\
                                             [0,2,3,10001],\
                                             [0,1,2,3]],dtype=np.int64)
```



- Other parameters



```python 
hbar,c,lambdax,epsilon0:
		numerical value of hbar,c, lambdax and epsilon0 in Flexible Unit (FU)  
        e.g. hbar, c, lambdax, epsilon0 = 1., 1., 1.6*10**28, 1.
```





- This function mainly determines the type of input parameters and helps the user to check them. If there is an error in the input parameter type then an error statement will be provided.

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
```

