---
title: Installation
author: Mingyan Sun 
date: 2022-04-01
layout: post
---



RGB-Maxwell已经上传至Github网站，用户通过git或者直接打开网站对代码进行下载即可；

```github
git clone https://github.com/KaimingHe/deep-residual-networks.git
```



RGB-Maxwell基于Python进行编写，在运行前需要保证用户安装以下程序包：

```python
import numba    ----numba是一款可以将python函数编译为机器代码的JIT编译器；
import math;    ----math提供了很多对浮点数的数学运算函数；
import cupy;    ----cuPy是NumPy兼容多维数组在CUDA上的实现;
import ray;     ----Ray是一种分布式执行框架;
import random;  ----random是一个能产生随机数的标准库；
import numpy;   ----NumPy是Python中科学计算的基础包；
import os;      ----os模块是Python标准库中的一个用于访问操作系统功能的模块；
import sys;     ----sys模块是一个用来处理python运行时环境的模块。
```



关于RGB-Maxwell的结构以及使用方法可以参考以下连接：

[RGB-Maxwell_Structure](http://127.0.0.1:4000/2022-06-29-General-Structure.html)

[RGB-Maxwell_Package](http://127.0.0.1:4000/jekyll/2022-07-02-Inner-package.html)

