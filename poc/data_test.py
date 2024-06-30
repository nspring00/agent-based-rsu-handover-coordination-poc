import io

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib

DATASET_PATH = '../datasets/vanet-trace-creteil-20130924-0700-0900/vanet-trace-creteil-20130924-0700-0900.csv'

MIN_X = 1131
MAX_X = 1330
MIN_Y = 511
MAX_Y = 710


def main():
    matplotlib.use('TkAgg')

    df = pd.read_csv(DATASET_PATH, sep=';')

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

    df["vehicle_x"] = df["vehicle_x"] - MIN_X
    df["vehicle_y"] = df["vehicle_y"] - MIN_Y

    grouped = df.groupby('timestep_time')

    save_grid(grid)
    save_background(grid)
    # draw_positions_timelapse(grouped, grid)
    # create_gif(grouped, grid)


def save_grid(grid):
    np.save('grid.npy', grid)


def save_background(grid):
    plt.figure(figsize=(4, 4))  # Set the figure size in inches
    plt.imshow(grid, cmap='gray')
    plt.axis('off')  # Turn off the axes
    plt.gca().invert_yaxis()
    plt.savefig('background.png', bbox_inches='tight', pad_inches=0, dpi=100)  # Save the figure as a jpg


def draw_positions_timelapse(grouped, grid):
    plt.figure(figsize=(10, 6))
    columns = ["vehicle_id", "vehicle_x", "vehicle_y", "vehicle_lane", "vehicle_speed", "vehicle_type"]
    for timestamp in grouped.groups.keys():
        data = grouped.get_group(timestamp)[columns].values

        # Clear the previous plot
        plt.clf()

        # Plot current positions
        plt.imshow(grid, cmap='gray')
        plt.scatter(data[:, 1], data[:, 2], color='red')
        plt.title(f'Vehicles at Timestamp: {timestamp}')
        plt.xlim(0, MAX_X - MIN_X)
        plt.ylim(0, MAX_Y - MIN_Y)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        # Display the plot
        plt.pause(0.2)  # Pause for a quarter of a second (adjustable)


def create_gif(grouped, grid):
    columns = ["vehicle_id", "vehicle_x", "vehicle_y", "vehicle_lane", "vehicle_speed", "vehicle_type"]

    frames = []  # List to hold the frames

    max_frames = min(1000, len(grouped))  # Limit to 300 frames
    print(f"Creating GIF with {max_frames} frames of {len(grouped)} frames.")

    timestamp_keys = list(grouped.groups.keys())[:max_frames]

    for timestamp in timestamp_keys:
        print(timestamp)
        data = grouped.get_group(timestamp)[columns].values

        plt.clf()  # Clear the figure

        plt.imshow(grid, cmap='gray')
        plt.scatter(data[:, 1], data[:, 2], color='red')
        plt.title(f'Vehicles at Timestamp: {timestamp}')
        plt.xlim(0, MAX_X - MIN_X)
        plt.ylim(0, MAX_Y - MIN_Y)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        # Instead of displaying the plot, save it to a buffer (in memory)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        frames.append(frame)
        buf.close()

    # Save the frames as a GIF
    imageio.mimsave('traffic_animation.gif', frames, fps=5)  # Adjust fps as needed


if __name__ == '__main__':
    main()
