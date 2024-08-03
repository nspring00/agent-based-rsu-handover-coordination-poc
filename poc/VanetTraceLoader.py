from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from matplotlib import pyplot as plt

from data_test import MIN_X, MAX_X, MIN_Y, MAX_Y

import numpy as np
import pandas as pd

DATASET_PATH = '../datasets/vanet-trace-creteil-20130924-0700-0900/vanet-trace-creteil-20130924-0700-0900.csv'


@dataclass
class VehicleTrace:
    id: str
    first_ts: int
    last_ts: int
    type: str
    trace: pd.DataFrame  # ts, x, y, angle, speed, lane


def map_trace(df: pd.DataFrame) -> Dict[str, VehicleTrace]:
    df = df.dropna(subset=['vehicle_x', 'vehicle_y'])
    df['timestep_time'] = df['timestep_time'].astype(int)

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


def get_traces() -> Dict[str, VehicleTrace]:
    if not Path('trace.npy').exists():
        trace = map_trace(pd.read_csv(DATASET_PATH, sep=';'))
        np.save('trace.npy', trace)
        return trace
    return np.load('trace.npy', allow_pickle=True).item()


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
    if not Path('grid.npy').exists():
        grid = map_grid(pd.read_csv(DATASET_PATH, sep=';'))
        np.save('grid.npy', grid)
        return grid
    return np.load('grid.npy')


def save_background(grid):
    plt.figure(figsize=(4, 4))  # Set the figure size in inches
    plt.imshow(grid, cmap='gray')
    plt.axis('off')  # Turn off the axes
    plt.gca().invert_yaxis()
    plt.savefig('background.png', bbox_inches='tight', pad_inches=0, dpi=100)  # Save the figure as a jpg


def plot_vehicle_count_per_timestep(vehicle_traces: Dict[str, VehicleTrace]):
    timestep_counts = defaultdict(int)

    for trace in vehicle_traces.values():
        timesteps = trace.trace["timestep_time"].values
        for timestep in timesteps:
            timestep_counts[timestep] += 1

    start_time = datetime.strptime("07:00:00", "%H:%M:%S")
    timesteps = sorted(timestep_counts.keys())
    times = [start_time + timedelta(seconds=int(t)) for t in timesteps]
    counts = [timestep_counts[t] for t in timesteps]

    experiment_start = datetime.strptime("07:15:00", "%H:%M:%S")
    experiment_end = datetime.strptime("08:45:00", "%H:%M:%S")
    experiment_counts = [count for time, count in zip(times, counts) if experiment_start <= time <= experiment_end]
    avg_vehicles = sum(experiment_counts) / len(experiment_counts) if experiment_counts else 0
    max_vehicles = max(experiment_counts) if experiment_counts else 0

    print(f"Average number of vehicles during experiment time: {avg_vehicles}")
    print(f"Maximum number of vehicles during experiment time: {max_vehicles}")

    plt.figure(figsize=(10, 6))
    plt.plot(times, counts, linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.title('Number of Vehicles over Time')
    plt.grid(True)

    # Add vertical lines at 7:15 AM and 8:45 AM
    # plt.axvline(x=datetime.strptime("07:15:00", "%H:%M:%S"), color='red', linestyle='--')
    # plt.axvline(x=datetime.strptime("08:45:00", "%H:%M:%S"), color='red', linestyle='--')

    # Add shaded region from 7:15 AM to 8:45 AM
    plt.axvspan(experiment_start, experiment_end, color='red', alpha=0.25)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    plt.savefig("Creteil_morning_vehicles_over_time.png", format="png", dpi=200)
    plt.show()


def main():
    # trace = map_trace(pd.read_csv(DATASET_PATH, sep=';'))
    # Save trace to file
    # np.save('trace.npy', trace)

    # trace = np.load('trace.npy', allow_pickle=True).item()

    traces = get_traces()
    plot_vehicle_count_per_timestep(traces)


if __name__ == "__main__":
    main()
