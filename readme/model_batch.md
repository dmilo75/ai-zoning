
# README for `model_batch.sbatch`

## Overview

This script is designed to be run as part of a Slurm job array. It automatically handles the rescheduling of the main python scrip QA_Code by waiting a rest period and then recalling the QA_Code until all question-municipality pairs are finished processing for a given specification.


## Key Variables and Parameters

- **`export_folder_name`**: The folder where all output and intermediate files are stored. This is provided as a command-line argument when the script is run.
- **`task_id`**: Automatically assigned based on the `SLURM_ARRAY_TASK_ID` environment variable, representing the node’s unique task in the job array. This allows for efficient parallel processing.
- **Status Check**: The `status_check_needed_{task_id}.txt` file is used to determine whether further job runs are required. This file is updated in the QA_Code itself but read and used by this script to determine if the job should be rescheduled.



## Script Workflow

1. **Task Assignment**: The script checks whether it's part of a Slurm job array, assigns a `task_id`, and prepares to run `QA_Code_V7.py` for the assigned task.
2. **Execution**: Runs the Python script (`QA_Code_V7.py`) inside a Singularity container with the specified export folder and task ID.
3. **Status Check**: After the Python script completes, the script checks whether further runs are required by examining the `status_check_needed_{task_id}.txt` file. If the file indicates that further processing is needed, the job will be rescheduled.
4. **Rescheduling**: If needed, the script reschedules itself to run again after 45 minutes using Slurm’s `sbatch` command.

## Execution

### Submitting the Script

To submit the job with Slurm, specify the export folder name:
```bash
sbatch --array=0-9 model_batch.sbatch <export_folder_name>
```

- The `--array=0-9` flag allows the script to run as a job array, with 10 tasks (0 through 9) processed in parallel. Adjust this based on the number of nodes you need.

### Singularity Execution

The script uses Singularity to manage the Python environment for `QA_Code_V7.py`. It binds directories to ensure proper access to files and sets up the environment with the following command:
```bash
singularity exec --nv --bind /vast/dm4766/cache:$HOME/.cache \
--overlay /scratch/dm4766/pytorch-example/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
/bin/bash -c "source /ext3/env.sh; python3 QA_Code_V7.py $export_folder_name $task_id"
```

## Rescheduling Logic

After each run, the script checks for the presence of the status check file:
```bash
status_check_file="${base_dir}/${export_folder_name}/status_check_needed_${task_id}.txt"
```

If the file exists and contains the value `True`, the job will be rescheduled to run again after 45 minutes using:
```bash
sbatch --begin=now+45minutes --array=${SLURM_ARRAY_TASK_ID} $0 "$1"
```

This ensures that the processing continues if there are additional tasks that need to be run.

