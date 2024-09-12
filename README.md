# Agent-based RSU Handover Coordination (ARHC) Strategy PoC

For more info on the functionality, see the [Master's Thesis](#) titled "Multi-Agent Systems in Vehicular Edge
Computing: A Communication-Criented Approach to Task Handovers".

## Installation

### Prerequisites

- Python 3.11+

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset used in the simulation is
the [Microscopic vehicular mobility trace of Europarc roundabout, Créteil, France](https://vehicular-mobility-trace.github.io/).

The dataset is not included in this repository, but can be
downloaded [here](https://vehicular-mobility-trace.github.io/#download).

Make sure to place both the morning and evening datasets in the `datasets` directory, i.e. the folder structure should
be as follows:

```
datasets
    morning
        vanet-trace-creteil-20130924-0700-0900
            vanet-trace-creteil-20130924-0700-0900.csv
    evening
        vanet-trace-creteil-20130924-1700-1900
            vanet-trace-creteil-20130924-1700-1900.csv
```

## Usage

The main runner is located in the `simulation_runner.py` file.

The ARHC strategy parameters tuning can be run using `eval_strategy_params()` and filling the search grid for the parameter evaluation.

Use `run_all_benchmarks()` to run all benchmarks in all configurations, or run individual configurations like `run_benchmarks("creteil-morning", "9-half")`.

The plots can be created using the methods in `utils/visualize_results.py`.
