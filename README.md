# README

This zip file contains reproductions of codes and results for 2020 and 2023 papers.

This dictionary is organized as follows:

```
.
├── 2020
│   ├── Phase Plane
│   │   ├── codes
│   │   │   ├── phase_plane.mlx
│   │   │   └── quiverplot.py
│   │   └── results
│   │       ├── WT.eps
│   │       ├── WT.mat
│   │       ├── all_over.eps
│   │       ├── all_over.mat
│   │       ├── hap4.eps
│   │       ├── hap4.mat
│   │       ├── no_mutual_inhibition.eps
│   │       ├── no_mutual_inhibition.mat
│   │       ├── sir2.eps
│   │       ├── sir2.mat
│   │       ├── sir_double.eps
│   │       └── sir_double.mat
│   └── Potential Landscape
│       ├── codes
│       │   ├── fokker-planck_2d.cu
│       │   ├── input_FP.par
│       │   └── plotsurf4.m
│       └── results
│           ├── all_over_labeled.eps
│           ├── n_00100_all_over.dat
│           ├── n_00100_sir_double.dat
│           ├── n_00100_wt.dat
│           ├── sir2_double_labeled.eps
│           └── wt_labeled.eps
└── 2023
    ├── DDE & Toogle Switch
    │   ├── codes
    │   │   ├── DDE_ts.mlx
    │   │   └── quiverplot.py
    │   └── results
    │       ├── os.eps
    │       ├── ts.eps
    │       └── ts.mat
    └── ODE
        └── ode.mlx
```

- 2020
  - Phase Plane: Deterministic modeling of Sir2-Hap4 systems.
  - Potential Landscape: Quasi-potential landscape computation reproduction.
- 2023
  - ODE: ODE modeling of synthetic oscillator.
  - DDE & Toogle Switch: DDE modeling of synthetic oscillator and modeling of toogle switch.