#!/bin/bash

# Define the log file directory
LOG_DIR="./logs"

# Check if log directory exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

# Function to run the experiment with specific configurations
run_experiment () {
    local representation=$1
    local k_edges=$2
    local normalized_metric=$3
    local NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    local LOG_FILE="$LOG_DIR/logging_${NOW}_rep_${representation}_k_${k_edges}_normalized_${normalized_metric}.log"
    local FLAG_FILE="$LOG_DIR/flag_$NOW.txt"

    rm -f "$FLAG_FILE"

    echo "Starting experiment with representation=$representation, k_edges=$k_edges, normalized_metric=$normalized_metric (Logging to $LOG_FILE)..."

    (
        # Run the command
        nohup python -u run_position_eap.py hydra.run.dir=. hydra/job_logging=disabled hydra/hydra_logging=disabled representation="$representation" k_edges="$k_edges" normalized_metric="$normalized_metric" > "$LOG_FILE" 2>&1

        # Create the flag file to signal completion
        echo "done" > "$FLAG_FILE"
    ) &

    # Wait for the flag file to be created
    while [ ! -f "$FLAG_FILE" ]; do
        sleep 1
    done

    # Optionally, remove the flag file after it's no longer needed
    rm -f "$FLAG_FILE"
    
    echo "Experiment with parameters: $representation, $k_edges, $normalized_metric completed."
}

# note: run experiments independent of the representation
run_experiment "words" "10" "False"
run_experiment "words" "20" "False"
run_experiment "words" "50" "False"
run_experiment "words" "100" "False"
run_experiment "words" "200" "False"
run_experiment "words" "10" "True"
run_experiment "words" "20" "True"
run_experiment "words" "50" "True"
run_experiment "words" "100" "True"
run_experiment "words" "200" "True"

