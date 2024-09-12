import itertools
import logging
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

from poc.model import VehicleAgent, VECModel, compute_vehicle_qos
from poc.render import VEC_STATION_COLORS
from poc.scenarios import SIMULATION_CONFIGS
from poc.strategies import DynamicVehicleLoadGenerator, STRATEGIES_DICT

SEED = 42

# Constants for the simulation, adjusted for demonstration
TIME_STEP_MS = 1000  # Time step in milliseconds
TIME_STEP_S = TIME_STEP_MS / 1000.0  # Time step in seconds
STEPS_PER_SECOND = int(1 // TIME_STEP_S)  # Number of steps per second
assert np.isclose(STEPS_PER_SECOND * TIME_STEP_S, 1.0, rtol=1e-09, atol=1e-09), "Time step conversion error"


def extract_model_metrics(model, model_name):
    """
    Prints the evaluation metrics for a given model.

    Parameters:
    - model: The model object to extract metrics from.
    - model_name: A string representing the name or identifier of the model.
    """
    df = model.datacollector.get_model_vars_dataframe()
    print(f"{model_name} Success: {df['TotalSuccessfulHandoverCount'].iloc[-1]}")
    print(f"{model_name} Failed: {df['TotalFailedHandoverCount'].iloc[-1]}")
    print(f"{model_name} QoS: {df['AvgQoS'].mean()}")
    print(f"{model_name} QoSMin: {df['MinQoS'].mean()}")
    print(f"{model_name} Gini: {df['GiniLoad'].mean()}")

    return [
        model_name,
        df['TotalSuccessfulHandoverCount'].iloc[-1],
        df['TotalSuccessfulHandoverCountRange'].iloc[-1],
        df['TotalSuccessfulHandoverCountLoadBalancing'].iloc[-1],
        df['TotalSuccessfulHandoverCountOverload'].iloc[-1],
        df['TotalFailedHandoverCount'].iloc[-1],
        df['AvgQoS'].mean(),
        df['AvgQoS'].std(),
        df['MinQoS'].mean(),
        df['MinQoS'].std(),
        df['AvgQoS_LoadBased'].mean(),
        df['AvgQoS_RangeBased'].mean(),
        df['GiniLoad'].mean(),
        df['GiniLoad'].std()
    ]


def run_model(params, max_steps=None):
    logging.disable(logging.CRITICAL)

    scenario, rsu_config_name, model_name, strategy_key, load_update_interval, seed, _, strategy_config = params
    strategy_class = STRATEGIES_DICT[strategy_key]

    if strategy_key == 'default':
        strategy = strategy_class(**strategy_config)
    else:
        strategy = strategy_class()

    trace_loader = SIMULATION_CONFIGS[scenario]["traces"]
    rsu_config = SIMULATION_CONFIGS[scenario][rsu_config_name]

    model = VECModel(strategy, rsu_config, DynamicVehicleLoadGenerator(seed=seed), trace_loader(), STEPS_PER_SECOND,
                     load_update_interval=load_update_interval, seed=seed)
    step = 0
    while model.running and (max_steps is None or step <= max_steps):
        model.step()
        step += 1

    filename = f"../results/runs/result_{scenario}_{rsu_config_name}_{model_name.lower()}"
    model.datacollector.get_model_vars_dataframe().to_csv(f"{filename}_model_vars.csv")
    # model.datacollector.get_agent_vars_dataframe().to_csv(f"{filename}_agent_vars.csv")

    return params, extract_model_metrics(model, model_name)


# Define parameter ranges for DefaultOffloadingStrategy
overload_threshold_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
leaving_threshold_values = [0]
imp_ho_timer_values = [0]
alt_ho_hysteresis_values = [0, 0.05, 0.1]
alt_suitability_min_values = [0.2, 0.25, 0.3, 0.35, 0.4]


def generate_default_strategy_configs(scenario, rsu_config):
    param_grid = itertools.product(
        overload_threshold_values,
        leaving_threshold_values,
        imp_ho_timer_values,
        alt_ho_hysteresis_values,
        alt_suitability_min_values
    )

    strategies = []
    for params in param_grid:
        config = {
            'overload_threshold': params[0],
            'leaving_threshold': params[1],
            'imp_ho_timer': params[2],
            'alt_ho_hysteresis': params[3],
            'alt_suitability_min': params[4]
        }
        name = f"Default_ovl{params[0]}_lvg{params[1]}_ho{params[2]}_hst{params[3]}_suit{params[4]}"
        strategies.append((scenario, rsu_config, name, 'default', 1, SEED, None, config))

    return strategies


BEST_ARHC_CONFIG = {
    'overload_threshold': 0.7,
    'leaving_threshold': 0,
    'alt_ho_hysteresis': 0.05,
    'alt_suitability_min': 0.3,
}


def store_results(results, filename):
    results.sort(key=lambda x: (not x[1][0].endswith('-Oracle'), x[1][0]))

    min_handovers = min(results, key=lambda x: x[1][1])[1][1]
    max_qos_mean = max(results, key=lambda x: x[1][6])[1][6]
    max_qos_min = max(results, key=lambda x: x[1][8])[1][8]
    min_gini = min(results, key=lambda x: x[1][12])[1][12]

    # Compute score of each entry by multiplying the % it achieves of the best score of each
    for _, result in results:
        # Compute the combined score for each entry; consider the difference between min and max values; should be <= 1
        ho_score = min_handovers / result[1]
        qos_mean_score = result[6] / max_qos_mean
        qos_min_score = result[8] / max_qos_min
        gini_score = min_gini / result[12]
        result.append(ho_score + qos_mean_score + qos_min_score + gini_score)
        result.append(ho_score * qos_mean_score * qos_min_score * gini_score)

    # Write to CSV with header line
    header = ("Model,HO_Total,HO_Range,HO_LB,HO_Overload,HO_Failed,AvgQoSMean,AvgQoSStd,MinQoSMean,MinQoSStd"
              ",AvgQoS_Load,AvgQoS_Range,GiniMean,GiniStd,EvalSum,EvalProd\n")
    header_len = len(header.split(","))
    with open(f"../results/{filename}.csv", "w") as f:
        f.write(header)
        for result in results:
            assert len(result[1]) == header_len, f"Length mismatch: {len(result[1])} vs {header_len}"
            f.write(",".join(map(str, result[1])) + "\n")


def create_run_model_with_steps(max_steps):
    def run_model_with_steps(params):
        return run_model(params, max_steps)

    return run_model_with_steps


def run_model_1000(params):
    return run_model(params, max_steps=1000)


def eval_strategy_params():
    start = time.time()

    scenario = "creteil-morning"
    rsu_config = "4-half"

    arhc_strategies = generate_arhc_strategy_configs(scenario, rsu_config)
    strategies = [
                     (scenario, rsu_config, "NearestRSU", "nearest", 1, SEED, 1388, None),
                     (scenario, rsu_config, "EarliestHO", "earliest", 1, SEED, 1540, None),
                     (scenario, rsu_config, "EarliestHONoBack", "earliest2", 1, SEED, 1494, None),
                     (scenario, rsu_config, "LatestHO", "latest", 1, SEED, 1264, None),
                 ] + arhc_strategies

    i = 0
    results = []
    if len(strategies) == 1:
        # Run in same thread for debugging
        results.append(run_model(strategies[0]))
    else:
        print("Start multi-threaded execution")
        with Pool(7) as p:
            # Run only 500 steps for param evaluation
            for res in p.imap_unordered(run_model, strategies):
                i += 1
                print(i, "/", len(strategies))
                results.append(res)

    print("Time elapsed:", int(time.time() - start), "s")

    store_results(results, f"results_eval_params_{scenario}_{rsu_config}")


def run_benchmarks(scenario, rsu_config):
    start = time.time()

    strategies = [
        (scenario, rsu_config, "ARHC-Oracle", "default", 0, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-01s", "default", 1, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-02s", "default", 2, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-03s", "default", 3, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-04s", "default", 4, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-05s", "default", 5, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-10s", "default", 10, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-15s", "default", 15, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-20s", "default", 20, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-25s", "default", 25, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "ARHC-30s", "default", 30, SEED, None, BEST_ARHC_CONFIG),
        (scenario, rsu_config, "NearestRSU", "nearest", 1, SEED, 1388, None),
        (scenario, rsu_config, "EarliestHO", "earliest", 1, SEED, 1540, None),
        # (scenario, rsu_config, "EarliestHONoBack", "earliest2", 1, SEED, 1494, None),
        (scenario, rsu_config, "LatestHO", "latest", 1, SEED, 1264, None),
    ]

    i = 0
    results = []
    if len(strategies) == 1:
        # Run in same thread for debugging
        results.append(run_model(strategies[0]))
    else:
        print("Start multi-threaded execution")
        with Pool(7) as p:
            for res in p.imap_unordered(run_model, strategies):
                i += 1
                print(i, "/", len(strategies))
                results.append(res)

    print("Time elapsed:", int(time.time() - start), "s")

    store_results(results, f"results_{scenario}_{rsu_config}")


def run_all_benchmarks():
    configs = [
        ("creteil-morning", "4-full"),
        ("creteil-morning", "4-half"),
        ("creteil-evening", "4-full"),
        ("creteil-evening", "4-half"),
        ("creteil-morning", "9-full"),
        ("creteil-morning", "9-half"),
        ("creteil-morning", "9-quarter"),
        ("creteil-evening", "9-full"),
        ("creteil-evening", "9-half"),
        ("creteil-evening", "9-quarter"),
        ("creteil-morning", "3-fail-full"),
        ("creteil-morning", "3-fail-half"),
        ("creteil-evening", "3-fail-full"),
        ("creteil-evening", "3-fail-half"),
    ]

    for scenario, rsu_config in configs:
        print(f"Running benchmarks for {scenario} with {rsu_config}")
        run_benchmarks(scenario, rsu_config)


def investigate_min_qos(trace, rsu_config_name, strategy):
    trace_loader = SIMULATION_CONFIGS[trace]["traces"]
    rsu_config = SIMULATION_CONFIGS[trace][rsu_config_name]
    model = VECModel(strategy, rsu_config, DynamicVehicleLoadGenerator(seed=SEED), trace_loader(),
                     load_update_interval=1, seed=SEED)

    grid_qos = defaultdict(list)

    step = 0
    while model.running:
        step += 1
        # if step > 500:
        #     break

        model.step()
        vehicles = model.schedule.get_agents_by_type(VehicleAgent)

        for vehicle in vehicles:
            grid_x = round(vehicle.pos[0])
            grid_y = round(vehicle.pos[1])
            grid_pos = (grid_x, grid_y)
            qos = compute_vehicle_qos(vehicle)
            grid_qos_list = grid_qos[grid_pos]
            grid_qos_list.append((step, qos))

    qos_mean_grid = np.full((200, 200), np.nan)
    qos_min_grid = np.full((200, 200), np.nan)

    for (x, y), qos_list in grid_qos.items():
        qos_mean_grid[y][x] = sum([q[1] for q in qos_list]) / len(qos_list)
        qos_min_grid[y][x] = min([q[1] for q in qos_list])

    filename = f"results_{trace}_{rsu_config_name}_heatmap_qos"
    np.save(filename + "_mean.npy", qos_mean_grid)
    np.save(filename + "_min.npy", qos_min_grid)
    model.datacollector.get_model_vars_dataframe().to_csv("model_vars.csv", index=False)

    plot_qos_grid(trace, rsu_config_name, filename + "_min.npy", min=True)


def plot_qos_grid(trace, rsu_config_name, filename='qos_grid.npy', min=True):
    qos_grid = np.load(filename)
    rsu_config = SIMULATION_CONFIGS[trace][rsu_config_name]

    reduction_factor = 4
    reduced_grid = np.zeros((qos_grid.shape[0] // reduction_factor, qos_grid.shape[1] // reduction_factor))
    for i in range(0, qos_grid.shape[0], reduction_factor):
        for j in range(0, qos_grid.shape[1], reduction_factor):
            slice_ = qos_grid[i:i + reduction_factor, j:j + reduction_factor]
            if np.isnan(slice_).all():
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nan
            else:
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nanmin(slice_)

    min_qos_min_value = 0.6 if np.nanmin(reduced_grid) >= 0.6 else 0.15
    assert np.nanmin(reduced_grid) >= min_qos_min_value, \
        f"Error: reduced_grid contains values below the threshold of {min_qos_min_value}."

    print(trace, rsu_config_name, "min MinQoS", np.nanmin(reduced_grid))

    # colors = [(1, 0, 0), (0.9, 0.9, 0.9)]  # Light gray to red
    # cmap = LinearSegmentedColormap.from_list('custom_gray_red', colors, N=256)

    cmap = "plasma"
    label = "Minimum QoS" if min else "Mean QoS"

    # Visualize the numpy array as a heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(reduced_grid, cmap=cmap, interpolation='nearest', vmin=min_qos_min_value, vmax=1)
    plt.colorbar(heatmap, label=label)
    # plt.title(label + " Heatmap")
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    for i, rsu in enumerate(rsu_config):
        rsu_id = 10001 + i
        pos = (rsu.pos[0] / reduction_factor, rsu.pos[1] / reduction_factor)
        color = VEC_STATION_COLORS[rsu_id]
        ax.add_patch(Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, facecolor=color))
        ax.add_patch(Circle(pos, rsu.range / reduction_factor, color=color, fill=False, linestyle='--', alpha=1))

    ax.set_aspect('equal')
    plt.tight_layout()
    filename = f"results_{trace}_{rsu_config_name}_{'min' if min else 'avg'}qos_heatmap.png"
    plt.savefig(filename, format="png", dpi=200)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # eval_strategy_params()
    # run_all_benchmarks()
    run_benchmarks("creteil-morning", "9-half")
    # investigate_min_qos("creteil-morning", "3-fail-half", DefaultOffloadingStrategy(**BEST_DEFAULT_CONFIG))
    # investigate_min_qos("creteil-morning", "3-fail-full", DefaultOffloadingStrategy(**BEST_DEFAULT_CONFIG))
    # plot_qos_grid("creteil-morning", "4-half", "results_creteil-morning_4-half_heatmap_qos_min.npy", min=True)
    # plot_qos_grid("creteil-morning", "9-quarter", "results_creteil-morning_9-quarter_heatmap_qos_min.npy", min=True)
    # plot_qos_grid("qos_grid_min.npy", "Minimum QoS", min=True)
    # plot_qos_versus_vehicle_count()
