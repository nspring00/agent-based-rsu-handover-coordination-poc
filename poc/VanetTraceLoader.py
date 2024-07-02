from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from functools import lru_cache

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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def get_grid():
    if not Path('grid.npy').exists():
        grid = map_grid(pd.read_csv(DATASET_PATH, sep=';'))
        np.save('grid.npy', grid)
        return grid
    return np.load('grid.npy')


def main():
    # trace = map_trace(pd.read_csv(DATASET_PATH, sep=';'))
    # Save trace to file
    # np.save('trace.npy', trace)

    trace = np.load('trace.npy', allow_pickle=True).item()


if __name__ == "__main__":
    main()
