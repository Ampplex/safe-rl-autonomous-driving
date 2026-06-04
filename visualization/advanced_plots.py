import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization.generate_plots import plot_performance_heatmap, plot_radar_chart


def generate_heatmap():
    os.makedirs("results/plots", exist_ok=True)
    plot_performance_heatmap()


def generate_radar_chart():
    os.makedirs("results/plots", exist_ok=True)
    plot_radar_chart()


if __name__ == "__main__":
    generate_heatmap()
    generate_radar_chart()
