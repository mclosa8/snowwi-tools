#!/bin/bash

usage="Usage: upload_flightline_to_bucket.sh [--dryrun] /base/path/YYYYMMDD/FLIGHTLINE date flightline aws_bucket aws_path"

# Parse --dryrun if present anywhere
dryrun=false
args=()

for arg in "$@"; do
    if [[ "$arg" == "--dryrun" ]]; then
        dryrun=true
    else
        args+=("$arg")
    fi
done

# Replace positional args with cleaned list
set -- "${args[@]}"

if [[ $# -ne 5 ]]; then
    echo "$usage"
    exit 1
fi

base_path=$(realpath "$1")
shift
date=$1
shift
flightline=$1
shift
bucket=$1
shift
aws_path=$1

# Build full path to radar data: base_path/date/flightline
path_to_radar_data="$base_path/$date/$flightline"

echo "Dry run mode: $dryrun"
echo "Base path: $base_path"
echo "Date: $date"
echo "Flightline: $flightline"
echo "Resolved radar data path: $path_to_radar_data"
echo "Bucket: $bucket"
echo "S3 prefix: $aws_path"

# Check radar data directory
if [[ ! -d "$path_to_radar_data" ]]; then
    echo "Error: Directory does not exist: $path_to_radar_data"
    exit 1
fi

# Check if S3 bucket is accessible
if ! aws s3 ls "s3://$bucket" > /dev/null 2>&1; then
    echo "Error: Bucket does not exist or you don't have access: $bucket"
    exit 1
else
    echo "Bucket $bucket is accessible."
fi

# Optional: Check if prefix exists
if ! aws s3 ls "s3://$bucket/$aws_path/" | grep -q .; then
    echo "Warning: Prefix 's3://$bucket/$aws_path/' does not exist. It will be created."
else
    echo "Prefix exists: s3://$bucket/$aws_path/"
fi

# Final S3 sync destination: bucket/aws_path/flightline/
s3_target_path="s3://$bucket/$aws_path/$flightline/"

echo "Preparing to sync: $path_to_radar_data → $s3_target_path"

# Build sync command
sync_command=(aws s3 sync "$path_to_radar_data" "$s3_target_path" --acl bucket-owner-full-control)

if $dryrun; then
    sync_command+=(--dryrun)
    echo "Dry run: No files will be uploaded."
fi

# Execute sync
"${sync_command[@]}"
status=$?

if [[ $status -ne 0 ]]; then
    echo "Error: Upload failed."
    exit 1
else
    echo $($dryrun && echo "Dry run complete." || echo "Upload complete.")
fi
