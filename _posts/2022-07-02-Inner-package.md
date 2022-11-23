---
title: --Inner Packages
author: Mingyan Sun 
date: 2022-07-02
category: Jekyll
layout: post
---

 这一部分主要介绍程序的主体结构。

程序由以下部分构成：
- Collision_database
    - Fusion_system
      - amplitude_square22
      - amplitude_square23
      - amplitude_square32
      - collision_type
    - QGP_system
      - amplitude_square22
      - amplitude_square23
      - amplitude_square32
      - collision_type
    - Test_system
      - amplitude_square22
      - amplitude_square23
      - amplitude_square32
      - collision_type
    - select_system
    - selected_system
- Collision_term
    - cuda_kernel22
    - cuda_kernel23
    - cuda_kernel32
    - main
- EMsolver
    - cuda_functions
    - region_distance
    - solver
- External_forces
    - cuda_kernel
    - main
- Macro_quantities
    - cuda_kernel
    - main
- Plasma
    - main
    - utils
- Plasma_methods
    - main

- Plasma_single_GPU
    - main
- Unit_conversion
    - main
- Vlasov_Drifit_terms
    - cuda_kernel
    - main 

