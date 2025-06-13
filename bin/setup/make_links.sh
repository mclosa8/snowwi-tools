#!/bin/bash

usage="make_links.sh /path/to/processing/root date flightline radar_data"

if [[ $# -ne 4 ]]
then
    echo $usage
    exit
fi

processing_root=$(realpath $1)
shift
date=$1
shift
flightline=$1
shift
radar_data=$1

echo $processing_root
echo $date
echo $flightline
echo $radar_data
echo

processing_dir="$processing_root/processing/$date/$flightline"

if [[ ! -d $processing_dir ]]
then
    echo "Invalid date or flightline."
    exit
fi

# Needed links: configs, patterns, radar_data

# Antenna patterns
patterns_dir="$processing_root/patterns"
if [[ ! -d $patterns_dir ]]
then
    echo "Invalid antenna patterns directory."
    exit
fi

ln -s "$patterns_dir" "$processing_dir/patterns"

# Radar data
radar_data_dir="$processing_root/radar_data/$date/$radar_data"
if [[ ! -d $radar_data_dir ]]
then
    echo "Invalid radar data directory."
    exit
fi
ln -s "$radar_data_dir" "$processing_dir/radar_data"

# Configs
# High
configs_high="$processing_root/configs/high"
if [[ ! -d $configs_high ]]
then
    echo "Invalid configs high directory."
    exit
fi

for i in {0..3}; do
  ln -s "$processing_dir/config/low/azmcomp_template.cfg" "$processing_dir/config/high/snowwi_azmcomp_$i.cfg"
done

# Low
configs_low="$processing_root/configs/low"
if [[ ! -d $configs_low ]]
then
    echo "Invalid configs low directory."
    exit
fi

for i in {0..3}; do
  ln -s "$processing_dir/config/low/azmcomp_template.cfg" "$processing_dir/config/low/snowwi_azmcomp_$i.cfg"
done

tree -L 3 $processing_dir
