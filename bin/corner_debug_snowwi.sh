#!/usr/bin/env bash
usage="corner_debug_snowwi.sh working_dir out_dir azmcomp_dir band corners.txt [ch [ch]]"
if [[ $# -lt 5 ]]
then
    echo "not enough arguments, usage is:"
    echo $usage
    exit 1
fi

SECONDS=0
date
if [[ $MINSAR_INIT_SCRIPT != '' ]]
then
    source $MINSAR_INIT_SCRIPT
fi
# set -o xtrace
set -o errexit
set -o pipefail
export PYTHONUNBUFFERED=1

working_dir="$1"
shift
corners_dir="$1"
shift
azmcomp_dir="$1"
shift
band="$1"
echo "Band is: $band"  # Debugging: Check the value of band
shift
debug_file="$(realpath $1)"
echo "Debug file: $debug_file"
shift
if [[ "$1" == '' ]]
then
    echo 'no ch'
    radar_channels=(0 1)
else
    echo 'ch'
    radar_channels=($@)
fi

echo $working_dir
working_dir=$(realpath $working_dir)
echo $working_dir
echo $corners_dir
echo $azmcomp_dir
echo $band
echo $debug_file
echo ${radar_channels[*]}

# echo "Contents of working dir: $(ls $working_dir)"

python -c 'import pycuda.autoinit as p; print("GPU:", p.device.name())'
error=$?
if [[ "$error" != '0' ]]
then
    echo "GPU error"
    exit
fi

cd $working_dir
if [[ ! -a "$corners_dir" ]]
then
    mkdir $corners_dir
fi
cd $corners_dir

run_channel() {
    echo "$#"
    cwd=$1
    azmcomp_dir=$2
    channel=$3
    band=$4
    debug_file=$5

    if [[ ! -d "$band" ]]; then
        echo "Making $band directory..."
        mkdir "$band"
    else
        echo "Directory $band already exists."
    fi
    cd "$band"
    mkdir "$channel"
    cd "$channel"
    if [[ ! -d "radar_data" ]]
    then
        echo "Link to radar_data directory not existing... Linking it..."
        ln -s ../../../radar_data/chan$channel radar_data
    else
        echo "Link to radar_data exists."
    fi

    config="snowwi_azmcomp_$channel.cfg"
    args="-b $band"

    echo "$args"
    echo "Config file: $config"
    echo "Band: $band"

    cp $cwd/${azmcomp_dir}/$band/$channel/*.json . # Copies radar state & antenna patterns
    cp $cwd/${azmcomp_dir}/${band}/$channel/*cfg . # Copies all configs

    setup_debug_snowwi.py $debug_file $config -u 2>&1 | tee setup_debug.log
    error=$?
    if [[ "$error" != '0' ]]
    then
        echo "Setup debug error."
        exit
    fi
    run_azmcomp.py -v -m "snowwi" --channel "$channel" \
        -c "$config" \
        --calibration-config "calibration_${channel}.cfg" \
        -n 25000 \
        -wd $cwd \
        -b "$band" \
        -d "target_indices.txt" 2>&1 | tee azmcomp.log

    read_debug.py azmcomp.h5 radar_state.json 2>&1 | tee read_debug.log
}
export -f run_channel


parallel -j 1 --halt now,fail=1 --line-buffer --gnu --link \
    run_channel $working_dir $azmcomp_dir {1} $band $debug_file ::: ${radar_channels[*]}

date
echo "elapsed: $SECONDS"
