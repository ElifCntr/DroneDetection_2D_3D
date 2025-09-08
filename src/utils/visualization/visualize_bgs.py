import yaml
import os
from utils.visualization.interactive_detect import test_bgs_interactive, run_interactive_detection

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Script directory: {script_dir}")

# Go up to project root (assuming structure: project_root/src/inference/visualize_bgs.py)
project_root = os.path.dirname(os.path.dirname(script_dir))
print(f"[DEBUG] Project root: {project_root}")

# Change to project root directory
os.chdir(project_root)
print(f"[DEBUG] Changed working directory to: {os.getcwd()}")

# Load config file
with open("configs/experiment.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Fix the config paths to match your YAML structure
test_videos = cfg["paths"]["input_video_dir"]  # "data/raw"
test_list = cfg["paths"]["splits"]["test_list"]  # "data/splits/test_videos.txt"

# Update the config to match what the interactive_detect expects
cfg["paths"]["raw_videos"] = cfg["paths"]["input_video_dir"]  # Add expected key
cfg["paths"]["annotations"] = cfg["paths"]["annotations_dir"]  # Add expected key

print(f"[INFO] Using videos from: {test_videos}")
print(f"[INFO] Using test list: {test_list}")
print(f"[INFO] Background method: {cfg['background']['method']}")

# Check if test list exists
if os.path.exists(test_list):
    print("[INFO] Starting interactive BGS testing with splits file...")
    run_interactive_detection(cfg, test_list)
elif os.path.exists(test_videos):
    print("[INFO] No splits file found, processing all videos in directory...")
    test_bgs_interactive(cfg, splits_file=None)
else:
    print("[ERROR] Neither splits file nor raw videos directory found!")
    print("Please check your paths in the config file.")