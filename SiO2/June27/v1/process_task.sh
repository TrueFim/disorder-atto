#!/bin/bash

# Script to process a single simulation task within a SLURM job array.

# Check if the parent directory argument is provided
if [ -z "$1" ]; then
  echo "Error: Parent directory of simulation folders not provided."
  echo "Usage: $0 /path/to/parent_simulation_directory"
  exit 1
fi

PARENT_DIR="$1"
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ -z "$TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. This script should be run as part of a SLURM job array."
  exit 1
fi

echo "--- Task Start ---"
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parent Simulation Directory: $PARENT_DIR"

# Get a sorted list of all simulation folders (directories) directly under PARENT_DIR
# The 'sort' command ensures a consistent mapping from TASK_ID to folder.
# If your folder names might contain spaces or unusual characters, this line might need adjustment.
SIM_FOLDERS=()
while IFS= read -r -d $'\0' dir; do
    SIM_FOLDERS+=("$dir")
done < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -not -name "slurm_logs" -print0 | sort -z)

# Check if any simulation folders were found
if [ ${#SIM_FOLDERS[@]} -eq 0 ]; then
  echo "Error: No simulation folders found in $PARENT_DIR."
  exit 1
fi

# Check if the TASK_ID is within the valid range
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "${#SIM_FOLDERS[@]}" ]; then
  echo "Error: Task ID $TASK_ID is out of bounds. Number of folders: ${#SIM_FOLDERS[@]}."
  echo "Folders found:"
  printf "  %s\n" "${SIM_FOLDERS[@]}"
  exit 1
fi

# SLURM_ARRAY_TASK_ID is 1-indexed. Bash arrays are 0-indexed.
TARGET_FOLDER_INDEX=$((TASK_ID - 1))
TARGET_FOLDER="${SIM_FOLDERS[$TARGET_FOLDER_INDEX]}"

if [ -z "$TARGET_FOLDER" ] || [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: Could not determine a valid target folder for Task ID $TASK_ID."
    echo "Attempted index: $TARGET_FOLDER_INDEX"
    echo "All found folders:"
    printf '%s\n' "${SIM_FOLDERS[@]}"
    exit 1
fi

echo "Processing folder: $TARGET_FOLDER"

# Change to the target folder
cd "$TARGET_FOLDER" || { echo "Error: Could not change directory to $TARGET_FOLDER"; exit 1; }

echo "Current directory: $(pwd)"

module purge
module load intel/2023.1.0.x
module load mkl/2023.1
module load impi/2021.9
module load cmake/3.30
module load anaconda/3/2023.03
module load libxc/6.2

libmkl_path=`find $INTEL_HOME -name libmkl_intel_lp64.so | head -n 1`
libmkl_path=`dirname "$libmkl_path"`
libintelc_path=`find $INTEL_HOME -name 'libintlc.so.5' | grep compiler/2024.0/lib/ | head -n 1`
libintelc_path=`dirname "$libintelc_path"`
export LD_LIBRARY_PATH="$libmkl_path":"$libintelc_path":$LIBXC_HOME/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

echo SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
cp ../Si_rps.dat .

srun -n $SLURM_JOB_NUM_NODES ~/local/bin/salmon < SiO2_gs_*.inp > OUTPUT_DFT.out
if [ $? -ne 0 -o ! -f "OUTPUT_DFT.out" ]; then
  echo "Error: the ground-state calculation failed."
  rm Si_rps.dat
  exit 1
fi

if grep -q "does not converged" OUTPUT_DFT.out; then
  echo "Error: the ground-state calculation failed to converge"
  rm Si_rps.dat
  exit 1
fi

mv data_for_restart restart
srun -n $SLURM_JOB_NUM_NODES ~/local/bin/salmon < SiO2_weak_rt_pulse_*.inp > OUTPUT_weak_pulse.out
srun -n $SLURM_JOB_NUM_NODES ~/local/bin/salmon < SiO2_rt_pulse_*.inp > OUTPUT_strong_pulse.out
srun -n $SLURM_JOB_NUM_NODES ~/local/bin/salmon < SiO2_response_rt_pulse_*.inp > OUTPUT_linear_response.out

rm Si_rps.dat
rm -rf restart

echo "--- Task $TASK_ID completed for folder $TARGET_FOLDER ---"
exit 0
