# Lattice reduction algorithms

This repository contains a Python-based framework demonstrating how lattice reduction algorithms improve lattice basis quality.

Detailed documentation available at: https://vttresearch.github.io/LatticeReductionAlgorithms/

Project repository is structured as follows:

```
├── bkz # python modules for lattice reduction algorithms
├── docs
├── LICENSE
├── main.py
├── Makefile
├── mkdocs.yml
├── plotter.py
├── README.md
├── requirements.txt
├── ruff.toml
└── tests
```

## Acknowledgements

The contents of this repository were implemented as part of Combinatorial Optimization with Hybrid Quantum-Classical Algorithms (COHQCA) -project funded by Business Finland. More information on the [project website](https://www.cohqca.fi/).

## License

Copyright © 2025 VTT Technological Research Centre of Finland Ltd. This repository is licensed under the terms and conditions described in [LICENSE](LICENSE).

## Contact

If you have questions regarding this repository you can contact markus.rautell@vtt.fi

## Getting started using Python virtual environment

1. Install python3-venv

```
sudo apt install python3-venv
```

2. Create virtual environment

```
python3 -m venv venv
```

3. Activate virtual environment

```
source venv/bin/activate
```

4. Install dependencies to your local virtual environment

```
pip install -r requirementx.txt
```

5. Now you're ready to start coding!

## Running the application

Run with the default parameters:

```
python3 main.py
```

Show the help message:

```
python3 main.py --help
```

You can specify parameters using the following arguments:

```
python3 main.py --lattice_dimension 10 --entry_bound 73 --bkz_version 1 --svp_solver 1 --block_size 5 --precision default --repetitions 5
```

# Running the test suite

You can run the existing tests with the following command

```
make test
```

OR manually using

```
pytest
```
