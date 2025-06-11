# Intra-Day Scheduling of Residential PV Battery Systems

[![](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-janik.pinter%40kit.edu-orange?label=Contact)](janik.pinter@kit.edu)

This repository contains the Python implementation for the master's thesis:
> [Multi-Objective Optimization for Intra-Day Scheduling of Residential PV Battery Systems] <br>
> Author: Paulina Hering

## Repository Structure
```
.
├── forecasting/ 
│   ├── create_quantile_forecasts.ipynb  # Create quantile prosumption forecasts based on real-world data
│   ├── convert_forecasts.ipynb          # Convert quantile forecasts to parametric form
│   ├── get_infinite_horizon_forecasts.ipynb # Convert quantile forecasts to parametric form for every starting hour
│   ├── visualize_forecasts.ipynb        # Visualize forecasts PDFs for every starting hour
│   └── visualize_data.ipynb             # Visualize forecasted PDFs
   
│
├── data/
│   ├── ground_truth/                    # Contains the ground-truth of prosumption of selected 
│   ├── electricity_costs                    # Contains electricity costs for selected time interval to include in future works  

real-world example
│   ├── quantile_forecasts/              # Contains the quantile forecasts
│   ├── parametric_forecasts/            # Contains parametric forecasts for gaussian distribution and every starting hour
│   └── parameters/                      # Contains json files with params such as cost-function weights, battery specifications, ...
│
└── optimization/ 
    ├── input_data.py                    # Load forecasts and parameters
    ├── experiment_tracking.py           # Tracks the experiment in MLFlow
    ├── optimization_model.py            # Contains the optimization problem
    ├── intraday_optimization_model.py   # Contains the optimization problem for intra-Day run
    ├── epsilon_const_optimization_model.py # Contains the optimization problem for Intra-Day in epsilon constraint form
    ├── intraday_solve.py                # Contains functions that handle the structure of Intra-Day Scheduling
    ├── visualize_costs.py               # Manually takes cost values and plots results
    ├── main.py                          # Executes the optimization problem
    ├── results_processing.py            # Visualizes the results
    ├── intraday_results_processing.py   # Visualizes results of Intra-Day run
    ├── pareto_front.py                  # Runs consecutive and MOO runs and plots pareto fronts
    ├── intraday_utils.py                # Contains utility functions for Intra-Day functionality
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

## Reproduce results
In order to reproduce the results shown in the paper, execute the optimization process with the corresponding parameter file. The necessary forecasts are included in the repository.<br>
To run Intra-Day Scheduling set intra_day_approach=True in main.py.
To run and plot 3D Pareto fronts set multiple_pareto_fronts=True in main.py
Both Intra-Day Scheduling and 3D Pareto fronts can be run for any resolution (time_slots=[4,10] corresponds to scheduling at 6am (Day-Ahead), 10am and 8pm. Can be up to time_slots=time_slots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], Be aware that the cost values are only sensible for an hourly resolution [1,2,3...,23])

Day 1, Day 2 and Day 3 in thesis correspond to timeframe1, timeframe2, timeframe3 in main.py.
The weights for MOO approach can be set as weights=[grid_weights, ss_weights] in line 67 in main.py


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



