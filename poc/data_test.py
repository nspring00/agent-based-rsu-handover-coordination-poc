import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_PATH = '../datasets/vanet-trace-creteil-20130924-0700-0900/vanet-trace-creteil-20130924-0700-0900.csv'

MIN_X = 1000
MAX_X = 1400
MIN_Y = 400
MAX_Y = 800


def main():
    # Read csv using pandas
    df = pd.read_csv(DATASET_PATH, sep=';')

    # Print the first 5 rows of the dataframe
    print(df.head())

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

    positions = set(map(tuple, positions))

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

    fig, ax = plt.subplots()

    ax.imshow(grid, cmap='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Roads')
    ax.invert_yaxis()

    plt.show()

    # Use matplotlib to visualize the grid
    plt.imshow(grid, cmap='gray')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()

    plt.gca().invert_yaxis()

    plt.show()


if __name__ == '__main__':
    main()
