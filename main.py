import json
import os
from analysis import ANALYSIS_ROUTER
import argparse

"""
python main.py --config examples/timeseries/lstm/config.json

"""


def main():
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description="Run SHAP Analysis")
    parser.add_argument('--config', type=str, help='Path to the config.json file')
    args = parser.parse_args()

    # 2. Use the provided path OR fall back to a default
    if args.config:
        config_path = args.config
    else:
        # Fallback: look in the same directory as main.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.json")

    # 3. Rest of your logic
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Find the analysis type if its tabular or timeseries
    analysis_type = config.get("analysis")

    # Route to the correct runner
    run_func = ANALYSIS_ROUTER.get(analysis_type)

    if run_func:
        run_func(config)
    else:
        print(f"Error: Analysis type '{analysis_type}' is not supported.")
        print(f"Supported types: {list(ANALYSIS_ROUTER.keys())}")


if __name__ == "__main__":
    main()


