#!/bin/bash

LANG=C.UTF-8

CUR_TIME=$(date +%Y%m%d-%H%M%S)
LOGDIR=~/log
LOGFILE=${LOGDIR}/gpu-$1-${CUR_TIME}.log

nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,name \
	--format=csv -l 1 \
	-f ${LOGFILE}

exit $?
