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


def main():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('../results/results.csv')

    # Sort the DataFrame by the 'Model' column alphabetically
    df = df.sort_values(by='Model')

    # Extract the relevant columns
    models = df['Model']
    successful = df['Successful']
    failed = df['Failed']

    # Create a bar chart
    x = range(len(models))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, successful, width, label='Successful')
    ax.bar([p + width for p in x], failed, width, label='Failed')

    # Add labels, title, and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('HO Count')
    ax.set_title('Successful and Failed Handovers per Model')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    # Display the plot
    plt.tight_layout()
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

    # Plot AvgQoSMean distribution
    plot_distribution(models, avg_qos_mean, avg_qos_std, 'Average QoS Mean Distribution', 'Density')
    # plot_beta_distribution(models, avg_qos_mean, avg_qos_std, 'Average QoS Mean Distribution', 'Density')

    # Plot MinQoSMean distribution
    plot_distribution(models, min_qos_mean, min_qos_std, 'Minimum QoS Mean Distribution', 'Density')
    # plot_beta_distribution(models, min_qos_mean, min_qos_std, 'Minimum QoS Mean Distribution', 'Density')

    # Plot GiniMean distribution
    plot_distribution(models, gini_mean, gini_std, 'Gini Mean Distribution', 'Density')
    # plot_beta_distribution(models, gini_mean, gini_std, 'Gini Mean Distribution', 'Density')


if __name__ == "__main__":
    main()
