#!/bin/bash

# Base folder containing room JSONs
BASE_FOLDER="$1"
SAVE_ROOT="$2"
MAIN_SCRIPT="$3"

if [[ -z "$BASE_FOLDER" || -z "$SAVE_ROOT" || -z "$MAIN_SCRIPT" ]]; then
    echo "Usage: $0 <base_folder> <save_root> <main_script.py>"
    exit 1
fi

mkdir -p "$SAVE_ROOT"

echo "=== Starting JSON processing ==="
echo "Base folder: $BASE_FOLDER"
echo "Save root: $SAVE_ROOT"
echo "Main script: $MAIN_SCRIPT"
echo "------------------------------------"

# Iterate over subfolders (room types)
for ROOM_DIR in "$BASE_FOLDER"/*; do
    if [[ -d "$ROOM_DIR" ]]; then
        ROOM_NAME=$(basename "$ROOM_DIR")
        
        # Iterate over JSON files
        for JSON_FILE in "$ROOM_DIR"/*.json; do
            if [[ -f "$JSON_FILE" ]]; then
                JSON_NAME=$(basename "$JSON_FILE" .json)
                SAVE_DIR="$SAVE_ROOT/${ROOM_NAME}_${JSON_NAME}"
                mkdir -p "$SAVE_DIR"

                echo "Processing JSON file: $JSON_FILE"
                echo "Room type: $ROOM_NAME"
                echo "Save directory: $SAVE_DIR"

                # Extract task description, assets list, and room dimensions
                IFS=$'\t' read -r TASK_DESC ASSETS DIM_X DIM_Y <<< "$(python3 - <<END
import json

scene_file = "$JSON_FILE"
with open(scene_file, 'r') as f:
    data = json.load(f)

task_desc = data.get("task_description", "This is an indoor room.")

room_dim = data.get("room_dimension", {})
dim_x = room_dim.get("room_dimension_x", 10)
dim_y = room_dim.get("room_dimension_y", 10)

assets = data.get("assets", {})
assets_list = [f"{v} {k}" for k, v in assets.items()]
assets_str = ", ".join(assets_list)

print(f"{task_desc}\t{assets_str}\t{dim_x}\t{dim_y}")
END
)"

                echo "Task description: $TASK_DESC"
                echo "Assets: $ASSETS"
                echo "Room dimensions: X=$DIM_X, Y=$DIM_Y"
                echo "------------------------------------"

                # Feed input line by line using a pipe
                {
                    echo "$SAVE_DIR"
                    echo "$TASK_DESC"
                    echo "$ASSETS"
                    echo "$DIM_X"
                    echo "$DIM_Y"
                    echo "quit"
                } | python3 -u "$MAIN_SCRIPT"

            fi
        done
    fi
done

echo "=== All JSONs processed ==="