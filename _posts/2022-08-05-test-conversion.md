---
title: Test Conversion
author: Mingyan Sun 
date: 2022-08-05
layout: post

---



This procedure primarily discusses the unit conversion within the RBG-Maxwell framework.

> For detailed information regarding the specific structure and content of the program, please refer to [Conversion](http://sunminmgyan.github.io/jekyll/2022-07-10-Conversion.html)



- Firstly, import the required packages for the program:

```python
from RBG_Maxwell.Unit_conversion.main import determine_coefficient_for_unit_conversion, unit_conversion
```



- Next, input the parameters to be converted.

  Here, the parameters are given in units of the International System of Units (SI).

```python
dx = dy = dz = 1000

# velocity
v_max = 1.8*10**6

# maximum momentum
momentum = 1.674*10**(-19)

# the momentum grid is set to be 
# half_px=half_pz=half_py=momentum
npx=npy=npz=20

dpx = dpz = dpy = 2*momentum/npy
dp_volume = dpx*dpy*dpz
dp = (dpx+dpy+dpz)/3

# time scale
dt = 10**(-6)

# number of maximum particles in each phase grid
n_max = 10000/(npx*npy*npz)

# number of averaged particles in each spatial grid
nx = ny = nz = 20
n_average = 10000/(nx*ny*nz)

E = 10**(-5)
B = 10**(-5)

masses = 9.3*10**(-26)
```



- To prevent the occurrence of excessively large values during the calculation process due to unit conversions, the program includes the "determine_coefficient_for_unit_conversion" function. This function ensures that the conversion process does not result in excessively large parameters. If the input parameters cannot be properly regulated to a lower level, the program will throw an error and suggest adjusting the input parameters.

```python
hbar, c, lambdax, epsilon0 = determine_coefficient_for_unit_conversion(dt, dx, dx*dy*dz, dp, dp_volume,n_max, n_average, v_max, E, B, masses, momentum )
```



- Then, when the parameters `hbar`, `c`, `lambdax`, and `epsilon0` are obtained, the program calls a function to establish a unit conversion table.
- Of course, users can also define the aforementioned parameters themselves and perform the unit conversion.

```python
conversion_table = \
unit_conversion('SI_to_LHQCD', coef_J_to_E=lambdax, hbar=hbar, c=c, k=1., epsilon0=epsilon0)

conversion_table_reverse = \
unit_conversion('LHQCD_to_SI', coef_J_to_E=lambdax, hbar=hbar, c=c, k=1., epsilon0=epsilon0)
```



- Finally, you can utilize the unit conversion table to perform unit conversions.

  SI units -> Free units

```python
meter_text = 200*conversion_table['meter']
masses_text = 9.3*10**(-26)*conversion_table['kilogram']
```



```python
meter_text : 47815189.40719678
masses_text : 2223415208.663215
```

​		 Free units -> SI units	

```python
meter_text = 47815189.40719678*conversion_table_reverse['TO_meter']
masses_text = 2223415208.663215*conversion_table_reverse['TO_kilogram']
```



```python
meter_text : 200
masses_text : 9.3*10**(-26)
```

