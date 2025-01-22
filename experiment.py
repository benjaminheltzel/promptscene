import os
from datetime import datetime

# Use this method throughout the notebook the get the newest folder path
def get_current_path(default_path=None):
    
    def is_valid_timestamp(timestamp: str) -> bool:
        """Checks if the given timestamp string matches the expected format."""
        try:
            datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
            return True
        except ValueError:
            return False
        
    def parse_timestamp(timestamp: str) -> datetime:
        """Parses a timestamp string into a datetime object."""
        return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")

    if not default_path:
        default_path = "experiments/merged_pipline"
    items = os.listdir(default_path)
    
    # Filter out items that match the 'run_{time_stamp}' pattern
    run_folders = [
        folder for folder in items
        if folder.startswith("run_") and is_valid_timestamp(folder[4:])
    ]
    
    if not run_folders:
        raise ValueError(f"No valid 'run_' folders found in {default_path}.")
    
    # Sort the folders by timestamp
    run_folders.sort(key=lambda folder: parse_timestamp(folder[4:]), reverse=True)
    
    # Return the absolute path to the newest folder
    return os.path.abspath(os.path.join(default_path, run_folders[0]))


# Setups the experiment by creating a new folder for each run
def setup_experiment(default_path=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
     
    if not default_path:
        default_path = "experiments/merged_pipline"
    
    output_path = os.path.join(default_path, f"run_{time_stamp}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Export path as variable to use it in bash scripts
    #os.environ['EXP_DIR'] = output_path
    
    print(f"Created new experiment folder: {output_path}")
    return output_path