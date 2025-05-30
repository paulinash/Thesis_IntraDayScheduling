# Probabilistic Day-Ahead Battery Scheduling based on Mixed Random Variables for Enhanced Grid Operation

[![](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-janik.pinter%40kit.edu-orange?label=Contact)](janik.pinter@kit.edu)

This repository contains the Python implementation for the paper:
> [Probabilistic Day-Ahead Battery Scheduling based on Mixed Random Variables for Enhanced Grid Operation](https://arxiv.org/abs/2411.12480) <br>
> Authors: Janik Pinter, Frederik Zahn, Maximilian Beichter, Ralf Mikut, and Veit Hagenmeyer

## Repository Structure
```
.
├── forecasting/ 
│   ├── create_quantile_forecasts.ipynb  # Create quantile prosumption forecasts based on real-world data
│   ├── convert_forecasts.ipynb          # Convert quantile forecasts to parametric form
│   └── visualize_data.ipynb             # Visualize forecasted PDFs
│
├── data/
│   ├── ground_truth/                    # Contains the ground-truth of prosumption of selected real-world example
│   ├── quantile_forecasts/              # Contains the quantile forecasts
│   ├── parametric_forecasts/            # Contains parametric forecasts for two distributions (Gaussian and sum of 2 logistic functions)
│   └── parameters/                      # Contains json files with params such as cost-function weights, battery specifications, ...
│
└── optimization/ 
    ├── input_data.py                    # Load forecasts and parameters
    ├── experiment_tracking.py           # Tracks the experiment in MLFlow
    ├── optimization_model.py            # Contains the optimization problem
    ├── main.py                          # Executes the optimization problem
    ├── results_processing.py            # Visualizes the results
    └── utils.py                         # Contains utility functions such as parametric pdf implementations

```

## Installation
1. Install virtualenv
   ```
   pip install virtualenv
   ```
2. Create a Virtual Environment
   ```
   virtualenv myenv
   ```
3. Activate the Virtual Environment
   ```
   source myenv/Scripts/activate
   ```
4. Install Packages specified in requirements-optimization.txt
   ```
   python -m pip install -r requirements-optimization.txt
   ```
Furthermore, ensure that IPOPT is properly installed. For more information, see
[IPOPT](https://github.com/coin-or/Ipopt)

## Execution
In order to start an optimization process, execute main.py.
   ```
   python optimization/main.py
   ```

## Reproducibility
In order to reproduce the results shown in the paper, execute the optimization process with the corresponding parameter file for Case 1, Case 2, or Case 3. The necessary forecasts are included in the repository.<br>
Note that for Case 1, uncertainties neither in the grid nor in the battery system are penalized. 
Thus, <ins>x</ins> and  ̅x are dispensable decision variables and are therefore set to zero (enable respective constraints in 'optimization_model.py' after '### Reproduce Case 1 ###' to obtain the same results as shown in the paper for Case 1).


In order to reproduce the forecasts, the following steps need to be done:
1. Install corresponding forecasting requirements
   ```
   python -m pip install -r requirements-forecasting.txt
   ```
2. Execute create_quantile_forecasts.ipynb with the following specifications:
    - The forecast were generated seeded using a system with the following specs, os, python version:
      - **Processor**: Intel 13th Gen Core i9-13900
      - **Memory**: 64 GB RAM
      - **Graphics**: NVIDIA GeForce RTX 3090 (Driver Version: 555.42.02 / CUDA Version: 12.5)
      - **OS**: Ubuntu 22.04.4 LTS
      - **PYTHON**: 3.12.5


## Funding
This project is funded by the Helmholtz Association under the "Energy System Design" program and the German Research Foundation as part of the Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation"

## License
This code is licensed under the [MIT License](LICENSE).
