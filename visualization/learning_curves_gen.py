import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp4_learning_curves import combine_curve_files


def generate_representative_curves():
    return combine_curve_files()


if __name__ == "__main__":
    generate_representative_curves()
