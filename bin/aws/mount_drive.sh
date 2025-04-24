#!/bin/bash

usage="mount_drive.sh /dev/drive /path/to/mountpoint"

if [[ $# -ne 2 ]]
then
    echo $usage
    exit
fi

drive="/dev/$1"
shift
dst=$(realpath $1)

echo $drive
echo $dst
echo

echo "Mounting $drive to $dst ..."

sudo file -s $dst

sudo mkfs -t xfs $drive

if [[ ! -d $dst ]]
then
    echo "Destination directory doesn't exist... Creating it."
    mkdir $dst
else
    echo "Destination directory already exists."
fi

sudo mount $drive $dst
sudo chown -R $USER:$USER $dst
df -h
