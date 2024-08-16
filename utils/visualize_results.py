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
    plt.title(f'Distribution of {title}')
    plt.xlabel(f'{ylabel} Value')
    plt.ylabel('Density')
    plt.legend(title="Models")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def custom_sort_key(model_name):
    if '-Oracle' in model_name:
        return 0, model_name
    return 1, model_name


def visualize_results(configs, experiment_title, plot_ho=True):
    configs = [(filename,
                pd.read_csv(f"../results/{filename}.csv").sort_values(by='Model', key=lambda x: x.map(custom_sort_key)),
                res_title)
               for filename, res_title in configs]

    if plot_ho:
        for filename, df, res_title in configs:
            plot_ho_count(filename, df, res_title)

    plot_metric(experiment_title, configs, 'GiniMean', 'Average Load Distribution Inequality (Gini Coefficient)',
                'Gini Coefficient',
                percentage=False, is_gini=True)
    plot_metric(experiment_title, configs, 'AvgQoSMean', 'Overall Network Performance (Average QoS)', 'Average QoS (%)',
                percentage=True)
    plot_metric(experiment_title, configs, 'MinQoSMean', 'Worst-Case Service Quality (Minimum QoS)', 'Minimum QoS (%)',
                percentage=True)


def plot_metric(experiment, configs, metric_col, title, ylabel, percentage=False, is_gini=False):
    plt.figure(figsize=(10, 5))

    colors = plt.get_cmap('tab10', 10)  # Get a colormap with at least 10 colors
    baseline_values = {}
    for i, (filename, df, res_title) in enumerate(configs):
        models = df['Model']
        metric_mean = df[metric_col]

        first_group = models[models.str.startswith('ARHC')]
        first_group_metric = metric_mean[:len(first_group)]

        last_group = models[len(first_group):]
        last_group_metric = metric_mean[len(first_group):]

        plt.plot(first_group, first_group_metric, label=f"ARHC - {res_title}", marker='o', color=colors(i))

        if is_gini:
            part_title = res_title.split(' ')[0]
            baseline_values[part_title] = (last_group, last_group_metric, f"Baseline - {part_title}")
        else:
            plt.scatter(last_group, last_group_metric, label=f"Baseline - {res_title}", marker='s',
                        color=colors(i))

    if is_gini:
        for j, (baseline_group, baseline_metric, baseline_title) in enumerate(baseline_values.values()):
            plt.scatter(baseline_group, baseline_metric, label=baseline_title, marker='s',
                        color=colors(len(configs) + j))

    legend_loc = "upper left" if is_gini else "lower left"

    plt.title(f'{experiment}: {title}')
    plt.xlabel('Handover Coordination Strategy')
    plt.ylabel(ylabel)
    if percentage:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.legend(title="Strategies & Configurations", loc=legend_loc)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f'{experiment.strip().lower().replace(" ", "_")}_{metric_col.lower().replace(" ", "_")}.png'
    plt.savefig(filename, format="png", dpi=200)
    plt.show()


def plot_ho_count(filename, df, title):
    models = df['Model']
    ho_range = df['HO_Range']
    ho_load_balancing = df['HO_LB']
    ho_overload = df['HO_Overload']
    successful = df['HO_Total']
    failed = df['HO_Failed']

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x, ho_range, width, label='Range HO', color='tab:blue')
    ax.bar(x, ho_load_balancing, width, bottom=ho_range, label='Load Balancing HO', color='tab:orange')
    ax.bar(x, ho_overload, width, bottom=ho_range + ho_load_balancing, label='Overload HO', color='tab:green')

    ax.bar([p + width for p in x], failed, width, label='Failed HO', color='tab:red')

    legend_loc = "lower right"
    # If max of last 3 bars is more than double of first, put the legend to upper left
    if np.max(successful[-3:]) > 2 * successful[0]:
        legend_loc = "upper left"

    ax.set_title(f'{title}: Successful and Failed Handovers')
    ax.set_xlabel('Handover Coordination Strategy')
    ax.set_ylabel('Number of Handovers')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title="Handover Type", loc=legend_loc)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{filename}_handovers.png', format="png", dpi=200)
    plt.show()


# Example configurations for Sparse and Dense scenarios
results_creteil_sparse = [
    ("results_creteil-morning_4-full", "Morning Full Capacity"),
    ("results_creteil-morning_4-half", "Morning Half Capacity"),
    ("results_creteil-evening_4-full", "Evening Full Capacity"),
    ("results_creteil-evening_4-half", "Evening Half Capacity"),
]

results_creteil_dense = [
    ("results_creteil-morning_9-full", "Morning Full Capacity"),
    ("results_creteil-morning_9-half", "Morning Half Capacity"),
    ("results_creteil-morning_9-quarter", "Morning Quarter Capacity"),
    ("results_creteil-evening_9-full", "Evening Full Capacity"),
    ("results_creteil-evening_9-half", "Evening Half Capacity"),
    ("results_creteil-evening_9-quarter", "Evening Quarter Capacity"),
]

results_creteil_dense_vs_sparse = [
    ("results_creteil-morning_4-full", "Sparse & Full Capacity"),
    ("results_creteil-morning_4-half", "Sparse & Half Capacity"),
    ("results_creteil-morning_9-full", "Dense & Full Capacity"),
    ("results_creteil-morning_9-half", "Dense & Half Capacity"),
    ("results_creteil-morning_9-quarter", "Dense & Quarter Capacity"),
]


# Main function to visualize results
def main():
    visualize_results(results_creteil_sparse, "Créteil Sparse")
    visualize_results(results_creteil_dense, "Créteil Dense")
    visualize_results(results_creteil_dense_vs_sparse, "Créteil Morning Sparse vs Dense", plot_ho=False)


if __name__ == "__main__":
    main()
