#!/bin/bash

usage='Usage:

get_flightline.sh [--dryrun] bucket prefix date flightline [path_to_dst]

NOTE: If [path_to_dst] not given, assumes cwd.'

# Parse arguments and detect --dryrun
dryrun_flag=""
args=()
for arg in "$@"; do
    if [[ "$arg" == "--dryrun" ]]; then
        dryrun_flag="--dryrun"
    else
        args+=("$arg")
    fi
done

if [[ ${#args[@]} -lt 4 ]]; then
    echo "$usage"
    exit 1
fi

bucket=${args[0]}
prefix=${args[1]}
date=${args[2]}
flightline=${args[3]}
path_to_dst=$(pwd -P)

if [[ ${#args[@]} -ge 5 ]]; then
    path_to_dst=$(realpath "${args[4]}")
fi

path_to_dst="$path_to_dst/$date/$flightline"

key="s3://$bucket/$prefix/$date/$flightline"

echo "Syncing: $key"
echo "To:      $path_to_dst"
[[ -n $dryrun_flag ]] && echo "(Dry run mode)"

aws s3 sync "$key" "$path_to_dst" $dryrun_flag
