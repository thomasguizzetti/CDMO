#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: ./script.sh SOLVER TIME_LIMIT MODEL OUTPUT_FILE INSTANCE_NAME"
    exit 1
fi

# Capture the arguments
SOLVER="$1"
TIME_LIMIT=300000
MODEL="$3"
OUTPUT_FILE="$4"
INSTANCE_NAME="$5"

# Specify the directory containing your instance files
INSTANCE_DIR="./instance_folder"

# Specify the full path to the instance
INSTANCE_PATH="$INSTANCE_DIR/$INSTANCE_NAME.dzn"

# Empty the output file if it exists
> "$OUTPUT_FILE"

# Extract the instance name without the extension
instance_name_no_ext="${INSTANCE_NAME%.dzn}"

# Print the instance name to the output file
echo "Running Instance: $instance_name_no_ext" >> "$OUTPUT_FILE"

# Print the command being executed
echo "Running command:"
echo "minizinc --solver $SOLVER --time-limit $TIME_LIMIT --output-time $MODEL.mzn $INSTANCE_PATH"

# Run the minizinc command for the specified instance and save the output to the txt file
minizinc --solver $SOLVER --time-limit $TIME_LIMIT \
         --output-time $MODEL.mzn $INSTANCE_PATH >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
