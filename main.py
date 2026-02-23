import json
import os
from analysis import ANALYSIS_ROUTER
import argparse

"""
python main.py --config examples/timeseries/lstm/config.json
python main.py --config examples/tabular/binary_classify/config.json
python main.py --config examples/tabular/multioutput_regress/config.json

"""


def main():
    # 1. Set up Argument Parser
    parser = argparse.ArgumentParser(description="Parse Explainer Config")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    # 2. Resolve the config path
    # If the user provides a path, we use it directly; otherwise, we look for config.json locally
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        print(f"Current Working Directory: {os.getcwd()}")
        return

    # 3. Load and Route
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {config_path} is not a valid JSON file.")
        return

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


