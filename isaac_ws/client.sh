#!/bin/bash

# Default value
TYPE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--multicube) TYPE="multicube"; shift ;;
        -s|--singlecube) TYPE="singlecube"; shift ;;
        -r|--real) TYPE="real_objects"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Check if type was set
if [[ -z "$TYPE" ]]; then
    echo "Error: You must specify either -m (multicube), -s (singlecube) or -r (real_objects)."
    exit 1
fi

# Run the actual Isaac Lab script with the chosen type
./isaac_ws/isaac_lab/isaaclab.sh -p isaac_ws/src/evaluate_openvla_"$TYPE".py --enable_cameras --save #--headless
