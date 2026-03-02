#!/bin/bash

usage() {
    echo "Usage: $0 --base_path|-bp <path> --out_path|-op <path> --first_file|-ff <n> --last_file|-lf <m>"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_path  | -bp)  BASE_PATH="$2";  shift 2 ;;
        --out_path   | -op)  OUT_PATH="$2";   shift 2 ;;
        --first_file | -ff)  FIRST_FILE="$2"; shift 2 ;;
        --last_file  | -lf)  LAST_FILE="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# Validate arguments
if [[ -z "$BASE_PATH" || -z "$OUT_PATH" || -z "$FIRST_FILE" || -z "$LAST_FILE" ]]; then
    echo "Error: all arguments are required."
    usage
fi

if [[ ! -d "$BASE_PATH" ]]; then
    echo "Error: base_path '$BASE_PATH' does not exist."
    exit 1
fi

# Copy files
for ch_dir in "$BASE_PATH"/ch*/; do
    ch=$(basename "$ch_dir")
    mkdir -p "$OUT_PATH/$ch"

    # Auto-detect filename base per channel
    first_file=$(ls "$ch_dir"*.dat 2>/dev/null | head -1)
    if [[ -z "$first_file" ]]; then
        echo "Warning: no .dat files found in '$ch_dir', skipping"
        continue
    fi
    FILENAME=$(basename "$first_file" | sed 's/_[0-9]*\.dat$//')
    echo "Channel '$ch': detected filename base '$FILENAME'"

    for ((i=FIRST_FILE; i<=LAST_FILE; i++)); do
        src_file="$ch_dir/${FILENAME}_${i}.dat"
        if [[ -f "$src_file" ]]; then
            cp "$src_file" "$OUT_PATH/$ch/"
        else
            echo "Warning: $src_file not found, skipping"
        fi
    done
done