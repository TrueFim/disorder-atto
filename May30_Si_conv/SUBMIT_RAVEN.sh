#!/bin/bash

# Main SLURM submission script for running an array of simulations.
# PLEASE REVISE "#SBATCH --array" BEFORE SUBMITTING THE JOB ARRAY!

# --- SLURM Directives ---
#SBATCH --array=1-6 # Here, 1 is the first index (not zero)!
#SBATCH --job-name=salmon
#SBATCH --chdir="./"
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL # Mail notifications
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm_logs/sim_array_%A_%a.out  # Standard output and error log (%A: JobID, %a: TaskID)
#SBATCH --error=slurm_logs/sim_array_%A_%a.err   # Standard error (explicitly, though often merged with output)


PARENT_SIM_DIR="./" # MUST BE THE SAME AS IN the `#SBATCH --chdir` DIRECTIVE ABOVE!

# Determine the absolute path to the parent simulation directory
# This makes it robust regardless of where sbatch executes the job.
ABS_PARENT_SIM_DIR=$(realpath "$PARENT_SIM_DIR")

if [ ! -d "$ABS_PARENT_SIM_DIR" ]; then
  echo "Error: Parent simulation directory '$PARENT_SIM_DIR' (resolved to '$ABS_PARENT_SIM_DIR') does not exist."
  exit 1
fi

# Count the number of simulation folders (direct subdirectories), excluding 'slurm_logs'
NUM_SIMULATIONS=0
# The find command below lists all directories one level deep, excluding any named "slurm_logs"
echo "Looking for simulation folders in: $ABS_PARENT_SIM_DIR, excluding 'slurm_logs'" # Added for clarity
while IFS= read -r -d $'\0' dir_path; do
    # echo "Found simulation candidate: $dir_path" # Uncomment for debugging
    ((NUM_SIMULATIONS++))
done < <(find "$ABS_PARENT_SIM_DIR" -mindepth 1 -maxdepth 1 -type d -not -name "slurm_logs" -print0)


if [ "$NUM_SIMULATIONS" -eq 0 ]; then
  echo "Error: No simulation subfolders (excluding 'slurm_logs') found in '$ABS_PARENT_SIM_DIR'."
  exit 1
fi

echo "Found $NUM_SIMULATIONS simulation folders (excluding 'slurm_logs') in $ABS_PARENT_SIM_DIR."

# --- Create log directory if it doesn't exist ---
mkdir -p slurm_logs

# --- Determine path to the task processing script ---
# Assuming process_task.sh is in the same directory as this script.
TASK_PROCESSOR_SCRIPT="./process_task.sh"

if [ ! -f "$TASK_PROCESSOR_SCRIPT" ]; then
    echo "Error: Task processor script '$TASK_PROCESSOR_SCRIPT' not found."
    exit 1
fi
if [ ! -x "$TASK_PROCESSOR_SCRIPT" ]; then
    echo "Error: Task processor script '$TASK_PROCESSOR_SCRIPT' is not executable. Please run 'chmod +x $TASK_PROCESSOR_SCRIPT'."
    exit 1
fi

echo "There are $NUM_SIMULATIONS simulation folders."
echo "Output/Error logs will be in the 'slurm_logs' directory."

# --- Command executed by SLURM for each array task ---
# This script will call `process_task.sh` and pass the absolute path
# to the parent directory of simulation folders.
"$TASK_PROCESSOR_SCRIPT" "$ABS_PARENT_SIM_DIR"
