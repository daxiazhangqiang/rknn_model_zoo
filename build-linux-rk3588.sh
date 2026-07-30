#!/bin/bash
if [ -z "$1" ]; then
    echo "missing specified model dir"
    exit -1
else
    echo "model=$1"
fi

# ./build-linux.sh -t rk3588 -a aarch64 -d "$1" -b Debug
./build-linux.sh -t rk3588 -a aarch64 -d "$1"