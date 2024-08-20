from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from matplotlib import pyplot as plt

from data_test import MIN_X, MAX_X, MIN_Y, MAX_Y

import numpy as np
import pandas as pd

DATASET_MORNING_PATH = '../datasets/vanet-trace-creteil-20130924-0700-0900/vanet-trace-creteil-20130924-0700-0900.csv'
DATASET_EVENING_PATH = '../datasets/vanet-trace-creteil-20130924-1700-1900/vanet-trace-creteil-20130924-1700-1900.csv'


@dataclass
class VehicleTrace:
    id: str
    first_ts: int
    last_ts: int
    type: str
    trace: pd.DataFrame  # ts, x, y, angle, speed, lane


def map_trace(df: pd.DataFrame, eval=True) -> Dict[str, VehicleTrace]:
    """
    Args:
        df: DataFrame containing the vehicle traces.
        eval: If True, load only the evaluation traces, otherwise load the full traces.
    """

    df = df.dropna(subset=['vehicle_x', 'vehicle_y'])
    df['timestep_time'] = df['timestep_time'].astype(int)

    if eval:
        # Timestep resembles seconds since start, so we can filter out the evaluation traces by checking if the timestep
        # is between 7:15 AM and 8:45 AM or 5:15 PM and 6:45 PM
        start_ts = 60 * 15  # 15 min
        end_ts = 60 * 105  # 105 min
        df = df[(df['timestep_time'] >= start_ts) & (df['timestep_time'] <= end_ts)]
        # Normalize the timestep to start from 0
        df['timestep_time'] = df['timestep_time'] - start_ts

    df = df[(df['vehicle_x'] >= MIN_X) & (df['vehicle_x'] <= MAX_X) & (df['vehicle_y'] >= MIN_Y) & (
            df['vehicle_y'] <= MAX_Y)]

    df['vehicle_x'] = df['vehicle_x'] - MIN_X
    df['vehicle_y'] = df['vehicle_y'] - MIN_Y

    df['vehicle_angle'] = df['vehicle_angle'] - 90

    # Sort the DataFrame by 'vehicle_id' and 'timestep_time'
    df = df.sort_values(by=['timestep_time', 'vehicle_id'])

    # Group the DataFrame by 'vehicle_id'
    grouped_df = df.groupby('vehicle_id')

    vehicle_traces = {}

    for vehicle_id, group in grouped_df:
        trace = group[['timestep_time', 'vehicle_x', 'vehicle_y', 'vehicle_angle', 'vehicle_speed', 'vehicle_lane']]
        vehicle_traces[vehicle_id] = VehicleTrace(vehicle_id, trace['timestep_time'].values[0],
                                                  trace['timestep_time'].values[-1], group['vehicle_type'].values[0],
                                                  trace)

    return vehicle_traces


def get_size() -> Tuple[int, int]:
    return MAX_X - MIN_X + 1, MAX_Y - MIN_Y + 1


def get_traces(morning=True, eval=True) -> Dict[str, VehicleTrace]:
    """
    Args:
        morning: If True, load the traces for the morning period, otherwise load the traces for the evening period.
        eval: If True, load only the evaluation traces, otherwise load the full traces.
    """

    filename = "../datasets/creteil_morning" if morning else "../datasets/creteil_evening"
    if eval:
        filename += "_eval"
    filename += "_trace.npy"

    dataset_path = DATASET_MORNING_PATH if morning else DATASET_EVENING_PATH

    if not Path(filename).exists():
        trace = map_trace(pd.read_csv(dataset_path, sep=';'), eval=eval)
        np.save(filename, trace)
        return trace
    return np.load(filename, allow_pickle=True).item()


def map_grid(df):
    df = df.dropna(subset=['vehicle_x', 'vehicle_y'])
    df = df[(df['vehicle_x'] >= MIN_X) & (df['vehicle_x'] <= MAX_X) & (df['vehicle_y'] >= MIN_Y) & (
            df['vehicle_y'] <= MAX_Y)]

    # I want to find out where roads are. Use the vehicles' x and y coordinates to find out.
    positions = df[['vehicle_x', 'vehicle_y']].values

    grid = np.ones((MAX_Y - MIN_Y + 1, MAX_X - MIN_X + 1))

    for x, y in positions:
        # Skip if x or y is nan
        if np.isnan(x) or np.isnan(y):
            continue
        grid[round(y) - MIN_Y, round(x) - MIN_X] = 0

    return grid


def get_grid():
    filename = "../datasets/creteil_grid.npy"
    if not Path(filename).exists():
        grid = map_grid(pd.read_csv(DATASET_MORNING_PATH, sep=';'))
        np.save(filename, grid)
        return grid
    return np.load(filename)


def save_background(grid):
    plt.figure(figsize=(4, 4))  # Set the figure size in inches
    plt.imshow(grid, cmap='gray')
    plt.axis('off')  # Turn off the axes
    plt.gca().invert_yaxis()
    plt.savefig('creteil_background.png', bbox_inches='tight', pad_inches=0, dpi=100)  # Save the figure as a jpg


def plot_vehicle_count_per_timestep(morning=True):
    vehicle_traces = get_traces(morning=morning, eval=False)
    timestep_counts = defaultdict(int)

    for trace in vehicle_traces.values():
        timesteps = trace.trace["timestep_time"].values
        for timestep in timesteps:
            timestep_counts[timestep] += 1

    start_time = datetime.strptime("07:00:00" if morning else "17:00:00", "%H:%M:%S")
    timesteps = sorted(timestep_counts.keys())
    times = [start_time + timedelta(seconds=int(t)) for t in timesteps]
    counts = [timestep_counts[t] for t in timesteps]

    experiment_start = datetime.strptime("07:15:00" if morning else "17:15:00", "%H:%M:%S")
    experiment_end = datetime.strptime("08:45:00" if morning else "18:45:00", "%H:%M:%S")
    experiment_counts = [count for time, count in zip(times, counts) if experiment_start <= time <= experiment_end]
    avg_vehicles = sum(experiment_counts) / len(experiment_counts) if experiment_counts else 0
    max_vehicles = max(experiment_counts) if experiment_counts else 0

    print(f"Average number of vehicles during experiment time: {avg_vehicles}")
    print(f"Maximum number of vehicles during experiment time: {max_vehicles}")

    plt.figure(figsize=(16, 6))
    plt.plot(times, counts, linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    title_start = 'Créteil Morning' if morning else 'Créteil Evening'
    plt.title(title_start + ' Trace: Number of Vehicles over Time')
    plt.grid(True)

    # Add vertical lines at 7:15 AM and 8:45 AM
    # plt.axvline(x=datetime.strptime("07:15:00", "%H:%M:%S"), color='red', linestyle='--')
    # plt.axvline(x=datetime.strptime("08:45:00", "%H:%M:%S"), color='red', linestyle='--')

    # Add shaded region from 7:15 AM to 8:45 AM
    plt.axvspan(experiment_start, experiment_end, color='red', alpha=0.25)

    # Add horizontal lines for average and maximum number of vehicles
    plt.axhline(y=avg_vehicles, color='blue', linestyle='--', label=f'Average: {avg_vehicles:.2f}')
    plt.axhline(y=max_vehicles, color='green', linestyle='--', label=f'Maximum: {max_vehicles}')

    plt.legend(loc='upper right')

    plt.tight_layout()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    filename = "creteil_morning_vehicles_over_time.png" if morning else "creteil_evening_vehicles_over_time.png"
    plt.savefig(filename, format="png", dpi=200)
    plt.show()


def plot_vehicle_count_per_timestep_full():
    plot_vehicle_count_per_timestep(morning=True)
    plot_vehicle_count_per_timestep(morning=False)


def plot_vehicle_positions_heatmap(morning=True):
    vehicle_traces = get_traces(morning=morning, eval=True)
    grid_position = defaultdict(int)

    for trace in vehicle_traces.values():
        positions = trace.trace[['vehicle_x', 'vehicle_y']].values
        for x, y in positions:
            grid_position[(round(x), round(y))] += 1

    vehicle_count_grid = np.full((200, 200), np.nan)

    for (x, y), count in grid_position.items():
        vehicle_count_grid[y, x] = count

    title = f"Vehicle Density Heatmap at Créteil Roundabout ({'Morning' if morning else "Evening"} Trace)"
    label = "Cumulative Vehicle Count"

    reduction_factor = 4
    reduced_grid = np.zeros(
        (vehicle_count_grid.shape[0] // reduction_factor, vehicle_count_grid.shape[1] // reduction_factor))

    for i in range(0, vehicle_count_grid.shape[0], reduction_factor):
        for j in range(0, vehicle_count_grid.shape[1], reduction_factor):
            slice_ = vehicle_count_grid[i:i + reduction_factor, j:j + reduction_factor]
            if np.isnan(slice_).all():
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nan
            else:
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nansum(slice_)

    # colors = [(0.9, 0.9, 0.9), (1, 0, 0)]  # Light gray to red
    # cmap = LinearSegmentedColormap.from_list('custom_gray_red', colors, N=256)

    fig, ax = plt.subplots()
    # heatmap = ax.imshow(reduced_grid, cmap="Wistia", interpolation='nearest')
    heatmap = ax.imshow(reduced_grid, cmap="plasma_r", interpolation='nearest')
    plt.colorbar(heatmap, label=label)
    plt.title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_aspect('equal')
    plt.tight_layout()

    filename = f"creteil_heatmap_{'morning' if morning else 'evening'}.png"
    plt.savefig(filename, format="png", dpi=200)
    plt.show()


def main():
    # trace = map_trace(pd.read_csv(DATASET_PATH, sep=';'))
    # Save trace to file
    # np.save('trace.npy', trace)

    # trace = np.load('trace.npy', allow_pickle=True).item()

    # plot_vehicle_count_per_timestep_full()
    plot_vehicle_positions_heatmap(True)
    plot_vehicle_positions_heatmap(False)


if __name__ == "__main__":
    main()
