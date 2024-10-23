#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=model  
#SBATCH --output=model_outfiles/%a_%A_model.out  # Dynamic output file name

# Check if the first command line argument is provided
if [ -z "$1" ]; then
  echo "Error: No export folder name provided."
  exit 1
fi

export_folder_name=$1
base_dir="/scratch/dm4766/ai-zoning/processed data/Model Output"  # Hardcoded base directory

# Dynamic adjustment of output file name if running as part of an array
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    export OUTPUT_FILE="model_outfiles/main_model_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    task_id=${SLURM_ARRAY_TASK_ID}
else
    export OUTPUT_FILE="model_outfiles/main_model_${SLURM_JOB_ID}.out"
    task_id=1
fi

echo "Running task with ID: $task_id"
echo "Output will be logged to: $OUTPUT_FILE"

module purge

mkdir -p model_outfiles

singularity exec --nv \
--bind /vast/dm4766/cache:$HOME/.cache \
--overlay /scratch/dm4766/pytorch-example/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
/bin/bash -c "source /ext3/env.sh; python3 QA_Code_V7.py $export_folder_name $task_id"

# Check exit status of the Python script
exit_status=$?

# Generate the path for the status check file
status_check_file="${base_dir}/${export_folder_name}/status_check_needed_${task_id}.txt"

echo "Checking for status check file: $status_check_file"

if [[ -e "$status_check_file" ]]; then
    status_needed=$(cat "$status_check_file")
    echo "Status check file found: $status_needed"

    if [[ "$status_needed" == "True" ]] && [[ $exit_status -eq 0 ]]; then
        echo "Rescheduling job as status is True"
        if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
            # Schedule the same sbatch file to run again after 1 hour for the same array task ID
            sbatch --begin=now+45minutes --array=${SLURM_ARRAY_TASK_ID} $0 "$1"
        else
            # Schedule the same sbatch file to run again after 1 hour for non-array jobs
            sbatch --begin=now+45minutes $0 "$1"
        fi
    else
        echo "No rescheduling needed as status is not True"
    fi
else
    echo "Status check file does not exist"
fi

# Print if Python script failed
if [[ $exit_status -ne 0 ]]; then
    echo "Python script execution failed with exit status: $exit_status"
fi
