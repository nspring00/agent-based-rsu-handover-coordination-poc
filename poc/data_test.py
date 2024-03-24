import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')

DATASET_PATH = '../datasets/vanet-trace-creteil-20130924-0700-0900/vanet-trace-creteil-20130924-0700-0900.csv'

MIN_X = 1131
MAX_X = 1330
MIN_Y = 511
MAX_Y = 710


def main():
    # Read csv using pandas
    df = pd.read_csv(DATASET_PATH, sep=';')

    # Print the first 5 rows of the dataframe
    # print(df.head())

    # Split the data by vehicle_id
    # vehicle_ids = df['vehicle_id'].unique()

    # print(len(vehicle_ids), vehicle_ids)
    # test_df = df[df['vehicle_id'] == vehicle_ids[0]]
    # # Print whole dataframe
    # print(test_df)

    df = df.dropna(subset=['vehicle_x', 'vehicle_y'])
    df = df[(df['vehicle_x'] >= MIN_X) & (df['vehicle_x'] <= MAX_X) & (df['vehicle_y'] >= MIN_Y) & (
            df['vehicle_y'] <= MAX_Y)]

    # I want to find out where roads are. Use the vehicles' x and y coordinates to find out.
    positions = df[['vehicle_x', 'vehicle_y']].values

    # print(len(positions))

    # positions = set(map(tuple, positions))

    # print(len(positions))

    # Get min and max x and y coordinates
    # min_x = min(x for x, y in positions)
    # max_x = max(x for x, y in positions)
    # min_y = min(y for x, y in positions)
    # max_y = max(y for x, y in positions)
    #
    # print(min_x, max_x, min_y, max_y)

    grid = np.ones((MAX_Y - MIN_Y + 1, MAX_X - MIN_X + 1))

    for x, y in positions:
        # Skip if x or y is nan
        if np.isnan(x) or np.isnan(y):
            continue
        grid[round(y) - MIN_Y, round(x) - MIN_X] = 0

    # print(*grid)

    print(grid.shape)

    # fig, ax = plt.subplots()
    #
    # ax.imshow(grid, cmap='gray')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Roads')
    # ax.invert_yaxis()
    #
    # plt.show()

    df["vehicle_x"] = df["vehicle_x"] - MIN_X
    df["vehicle_y"] = df["vehicle_y"] - MIN_Y

    grouped = df.groupby('timestep_time')

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


if __name__ == '__main__':
    main()
