# main.py
import json
import os
from analysis import ANALYSIS_ROUTER

def main():
    # 1. Get the directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Point to the config file in that same directory
    config_path = os.path.join(base_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        # Print the current working directory to help debug
        print(f"Current Working Directory: {os.getcwd()}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Find the high-level category (e.g., "timeseries")
    analysis_type = config.get("analysis")
    
    # 3. Route to the correct sub-folder's runner
    run_func = ANALYSIS_ROUTER.get(analysis_type)
    
    if run_func:
        print(f"--- Starting {analysis_type} Analysis ---")
        run_func(config)
    else:
        print(f"Error: Analysis type '{analysis_type}' is not supported.")

if __name__ == "__main__":
    main()