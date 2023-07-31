---
title: Installation
author: Mingyan Sun 
date: 2022-04-01
layout: post
---



The **RBG-Maxwell framework** has been uploaded to the GitHub. Users can download the source code via **git**.

```github
git clone https://github.com/sunminmgyan/sunminmgyan.github.io.git
```

**RBG-Maxwell** is written in **python**, so the user needs to ensure that the following packages are installed before running the program

```python
import numba   
	----numba is a JIT compiler that can compile python functions into machine code；
import math;    
	----math provides a number of mathematical functions for floating point numbers；
import cupy;   
	----cupy is an implementation of NumPy-compatible multidimensional arrays on CUDA;
import ray;     
	----Ray is a distributed execution framework;
import random;  
	----random is a standard library that generates random numbers；
import numpy;  
	----NumPy is the base package for scientific computing in Python；
import os;      
	----os is a module in the Python standard library for accessing operating system functions；
import sys;     
	----sys is a module to handle the python runtime environment。
```



The structure of the RBG-Maxwell framework can be found at [General Structure](https://sunminmgyan.github.io/pages/2022-06-01-General-Structure/)



