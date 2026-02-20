import json
import os
from analysis import ANALYSIS_ROUTER


def main():
    # Get the directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        print(f"Current Working Directory: {os.getcwd()}")
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


