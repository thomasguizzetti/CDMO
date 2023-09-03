#!/bin/bash

# Specify the directory containing your instance files
INSTANCE_DIR="instance_folder"

# Specify the solver and other options
SOLVER="gecode"
TIME_LIMIT="300000"

# Loop over all model files in the directory
for model_file in model_folder/*.mzn; do
    # Extract the model file name without the path
    model_name=$(basename "$model_file")
    
    # Extract the model name without the extension
    model_name_no_ext="${model_name%.mzn}"
    
    # Construct the output file name
    output_file="${model_name_no_ext}_${SOLVER}.txt"

    # Print the header to the output file
    echo "Solver: $SOLVER" >> "$output_file"
    echo "Model: $model_name_no_ext" >> "$output_file"
    echo "" >> "$output_file"

    # Loop over all instance files in the directory
    for instance in "$INSTANCE_DIR"/*.dzn; do
        # Extract the instance file name without the path
        instance_name=$(basename "$instance")
        
        # Extract the instance name without the extension
        instance_name_no_ext="${instance_name%.dzn}"
        
        # Print the instance name to the output file
        echo "Running Instance: $instance_name_no_ext" >> "$output_file"
        
        # Print the command being executed
        echo "Running command:"
        echo "minizinc --solver $SOLVER --time-limit $TIME_LIMIT $model_file $instance"
        echo "Running Instance: $instance_name"

        # Record the start time
        start_time=$(date +%s.%N)
        
        # Run the minizinc command for each instance and save the output to the txt file
        minizinc --solver $SOLVER --time-limit $TIME_LIMIT "$model_file" "$instance" >> "$output_file"
        
        # Calculate and print the time taken
        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)
        echo "Time taken: $elapsed_time seconds" >> "$output_file"

        echo "" >> "$output_file"
    done
done
