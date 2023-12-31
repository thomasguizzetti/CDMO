#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: ./script.sh SOLVER MODEL OUTPUT_FILE INSTANCE_NAME"
    exit 1
fi

TIME_LIMIT=300000

# Capture the arguments
SOLVER="$1"
MODEL="$2"
OUTPUT_FILE="$3"
INSTANCE_NAME="$4"

# Specify the directory containing your instance files
INSTANCE_DIR="./instance_folder"
MODEL_DIR="./model_folder"

# Specify the full path to the instance
INSTANCE_PATH="$INSTANCE_DIR/$INSTANCE_NAME.dzn"
MODEL_PATH="$MODEL_DIR/$MODEL.mzn"


# Empty the output file if it exists
> "$OUTPUT_FILE"

# Extract the instance name without the extension
instance_name_no_ext="${INSTANCE_NAME%.dzn}"

# Print the instance name to the output file
echo "Running Instance: $instance_name_no_ext" >> "$OUTPUT_FILE"

# Print the command being executed
echo "Running command:"
echo "minizinc --solver $SOLVER --time-limit $TIME_LIMIT --output-time $MODEL_PATH $INSTANCE_PATH"

# Run the minizinc command for the specified instance and save the output to the txt file
minizinc --solver $SOLVER --time-limit $TIME_LIMIT \
         --output-time $MODEL_PATH $INSTANCE_PATH >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
