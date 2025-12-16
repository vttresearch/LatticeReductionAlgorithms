# main

Description of `main.py`-module. This is the main entry point to the application.

Usage:

```
usage: main.py [-h] [--lattice_dimension LATTICE_DIMENSION] [--entry_bound ENTRY_BOUND] [--bkz_version {1,2,3}] [--svp_solver {1,2,3}] [--block_size BLOCK_SIZE] [--precision PRECISION]
               [--repetitions REPETITIONS]

Run lattice reduction algorithms.

options:
  -h, --help            show this help message and exit
  --lattice_dimension LATTICE_DIMENSION
                        Desired lattice dimension. (default: 10)
  --entry_bound ENTRY_BOUND
                        Bound for basis entry values (default: 73)
  --bkz_version {1,2,3}
                        Specify the version of bkz implementation: 1: bkz_se, 2: bkz_se_bfp_track, 3: bkz_se_sum_track (default: 1)
  --svp_solver {1,2,3}  Specify the svp_solver utilized during bkz execution: 1: enum_se_og_solver, 2: enum_se_solver, 3: enum_sh_solver (default: 1)
  --block_size BLOCK_SIZE
                        Desired block size for bkz. (default: 5)
  --precision PRECISION
                        Precision of floating point arithmetic: high, default, low. (default: default)
  --repetitions REPETITIONS
                        Number of random bases to operate on. (default: 5)

```

::: main
    options:
        show_source: false

