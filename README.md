# Residential Day-Ahead PV-Battery Scheduling based on Mixed Random Variables for Enhanced Grid Operations

[![](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-janik.pinter%40kit.edu-orange?label=Contact)](janik.pinter@kit.edu)

ToDo: Insert paper/Author information

## Repository Structure
The repository is structured as shown below:
```
.
├── forecasting/ 
│   ├── create_quantile_forecasts.ipynb  # Based on real-world data, create quantile prosumption forecasts
│   ├── convert_forecasts.ipynb          # Convert quantile forecasts to parametric form and store results
│   └── visualize_data.ipynb             # Visualize forecasted PDFs
│
├── data/
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
Todo: Add step by step guide

## Execution
Todo: Add Step by step guide

## Funding
This project is funded by the Helmholtz Association under the "Energy System Design" program and the German Research Foundation as part of the Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation"

## License
This code is licensed under the [MIT License](LICENSE).
