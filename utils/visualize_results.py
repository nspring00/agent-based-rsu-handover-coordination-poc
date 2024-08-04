import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution(models, means, stds, title, ylabel):
    plt.figure(figsize=(10, 5))
    colors = plt.get_cmap('tab10', len(models))
    for i, (model, mean, std) in enumerate(zip(models, means, stds)):
        x = np.linspace(max(0, mean - 3 * std), min(1, mean + 3 * std, 100))
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        plt.plot(x, y, label=model, color=colors(i))
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_results(configs, title):
    configs = [(filename, pd.read_csv(f"../results/{filename}.csv").sort_values(by='Model'), title) for filename, title in
               configs]

    for filename, df, res_title in configs:
        plot_ho_count(filename, df, res_title)

    plot_metric(configs, 'GiniMean', f'{title}: Average Gini Values for Different Models', 'Gini Value')
    plot_metric(configs, 'AvgQoSMean', f'{title}: Average QoS Values for Different Models', 'Avg QoS Value',
                qos=True)
    plot_metric(configs, 'MinQoSMean', f'{title}: Minimum QoS Values for Different Models', 'Min QoS Value',
                qos=True)


def plot_metric(configs, metric_col, title, ylabel, qos=False):
    plt.figure(figsize=(10, 5))

    for i, (filename, df, res_title) in enumerate(configs):
        models = df['Model']
        metric_mean = df[metric_col]

        first_group = models[models.str.startswith('DefaultShare')]
        first_group_metric = metric_mean[:len(first_group)]

        last_group = models[len(first_group):]
        last_group_metric = metric_mean[len(first_group):]

        plt.plot(first_group, first_group_metric, label=f"Proposed Strategy - {res_title}", marker='o')

        if not qos and i == 0:
            plt.scatter(last_group, last_group_metric, label="Baseline Strategies", marker='s', color='red')

        if qos:
            plt.scatter(last_group, last_group_metric, label=f"Baseline Strategies - {res_title}", marker='s')

    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.tight_layout()
    plt.show()


def plot_ho_count(filename, df, title):
    # Read the CSV file into a DataFrame
    # df = pd.read_csv(filename)
    #
    # # Sort the DataFrame by the 'Model' column alphabetically
    # df = df.sort_values(by='Model')

    # Extract the relevant columns
    models = df['Model']
    ho_range = df['HO_Range']
    ho_load_balancing = df['HO_LB']
    ho_overload = df['HO_Overload']

    successful = df['HO_Total']
    failed = df['HO_Failed']

    # Create a bar chart
    x = range(len(models))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()

    # Create stacked bars for successful handovers
    ax.bar(x, ho_range, width, label='Range HO')
    ax.bar(x, ho_load_balancing, width, bottom=ho_range, label='Load Balancing HO')
    ax.bar(x, ho_overload, width, bottom=ho_range + ho_load_balancing, label='Overload HO')

    # Create bars for failed handovers
    ax.bar([p + width for p in x], failed, width, label='Failed HO')

    # Add labels, title, and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('HO Count')
    ax.set_title(title + ': Successful and Failed Handovers per Model')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    # Display the plot
    plt.tight_layout()

    filename = filename + '_handovers.png'
    plt.savefig(filename, format="png", dpi=200)

    plt.show()

    # Plot distributions
    # Extract the relevant columns
    models = df['Model']
    avg_qos_mean = df['AvgQoSMean']
    avg_qos_std = df['AvgQoSStd']
    min_qos_mean = df['MinQoSMean']
    min_qos_std = df['MinQoSStd']
    gini_mean = df['GiniMean']
    gini_std = df['GiniStd']

    return

    # Plot AvgQoSMean distribution
    plot_distribution(models, avg_qos_mean, avg_qos_std, title + ': Average QoS Mean Distribution', 'Density')
    # plot_beta_distribution(models, avg_qos_mean, avg_qos_std, 'Average QoS Mean Distribution', 'Density')

    # Plot MinQoSMean distribution
    plot_distribution(models, min_qos_mean, min_qos_std, title + ': Minimum QoS Mean Distribution', 'Density')
    # plot_beta_distribution(models, min_qos_mean, min_qos_std, 'Minimum QoS Mean Distribution', 'Density')

    # Plot GiniMean distribution
    plot_distribution(models, gini_mean, gini_std, title + ': Gini Mean Distribution', 'Density')
    # plot_beta_distribution(models, gini_mean, gini_std, 'Gini Mean Distribution', 'Density')


results_creteil_morning_sparse = [
    # ("results", "Demo")
    ("results_creteil-morning_4-full", "Full Capacity"),
    ("results_creteil-morning_4-half", "Half Capacity"),
]


def main():
    visualize_results(results_creteil_morning_sparse, "Creteil Morning Sparse")


if __name__ == "__main__":
    main()
