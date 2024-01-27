#!/bin/bash

# Copyright 2024 NOGUCHI, Shoji
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LANG=C.UTF-8

function usage_exit() {
    echo "Usage: $0 [-i gpu_id(s)] [-d directory] <log-prefix>" 1>&2
    exit 1
}

# Function to display string
function puts() {
    printf '%s\n' "$*"
}

# Function to display commands
function puts_exec() {
    puts "Execute commands: $*"
    eval "$@" 2>&1
}

#
# analyze options.
#
while getopts d:hi: OPT; do
    case $OPT in
        d)  LOGDIR=${OPTARG}
            ;;
        h)  usage_exit
            ;;
        i)  GPU_IDS=${OPTARG}
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
    usage_exit
fi

LOG_PREFIX=$1

: ${LOGDIR:="~/log"}
if [[ ! -d ${LOGDIR} ]]; then
    puts "Error: log directory not found: ${LOGDIR}"
    exit 1
fi

NVIDIA_SMI_OPTS=""
if [[ ! -z ${GPU_IDS} ]]; then
    NVIDIA_SMI_OPTS="--id=${GPU_IDS}"
fi

CUR_TIME=$(date +%Y%m%d-%H%M%S)
LOGFILE=${LOGDIR}/gpu-${LOG_PREFIX}-${CUR_TIME}.log

puts_exec nvidia-smi \
    --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,name \
	--format=csv \
    -l 1 \
	-f ${LOGFILE} \
    ${NVIDIA_SMI_OPTS}

exit $?
