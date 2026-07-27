#!/bin/bash
if [ -z "$1" ]; then
    # per layer performance log level set
    export RKNN_LOG_LEVEL=4
else
    # per frame performance log level set
    export RKNN_LOG_LEVEL=$1
fi
    echo "RKNN_LOG_LEVEL=${RKNN_LOG_LEVEL}"
