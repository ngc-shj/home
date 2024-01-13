#!/bin/bash

LANG=C.UTF-8

function usage_exit() {
	echo "Usage: $0 [-o OUTDIR] CSV_FILE" 1>&2
	exit 1
}

# Function to dispalay string
function puts() {
	printf '%s\n' "$*"
}

#
# analyze options.
#
while getopts ho: OPT; do
	case $OPT in
		h)  usage_exit
			;;
		o)  OUT_DIR=${OPTARG}
			;;
		\?) usage_exit
			;;
	esac
done

shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
	usage_exit
fi

# * e.g. Format of input csv file.
# ---
# index, timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.free [MiB], name
# 0, 2024/01/12 11:11:57.087, 2 %, 7 %, 604 MiB, 15459 MiB, NVIDIA GeForce RTX 4090 Laptop GPU
# 1, 2024/01/12 11:11:57.091, 0 %, 0 %, 0 MiB, 24156 MiB, NVIDIA GeForce RTX 4090
# 0, 2024/01/12 11:11:58.096, 0 %, 13 %, 604 MiB, 15459 MiB, NVIDIA GeForce RTX 4090 Laptop GPU
# 1, 2024/01/12 11:11:58.098, 0 %, 0 %, 0 MiB, 24156 MiB, NVIDIA GeForce RTX 4090
# ...
#
# * e.g. Command line
# $ nvidia-smi --query-gpu=index,name,timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu --format=csv,nounits -l 1 -f temp.csv &
#
INPUT_CSV=$1
if [[ ! -f ${INPUT_CSV} ]]; then
	puts "Error: CSV file not found: ${INPUT_CSV}"
	exit 1
fi

: ${OUT_DIR:="./"}
if [[ ! -d ${OUT_DIR} ]]; then
	puts "Error: Directory not found: ${OUT_DIR}"
	exit 1
fi

#
# Analyze header in csv file
#
declare -A DATA_IDX DATA_UNIT
IFS=',' read -r -a COLUMNS <<< "$(head -1 ${INPUT_CSV})"

idx=1
for column in "${COLUMNS[@]}"; do
	# triming
	column=${column%* }
	column=${column# *}
	#
	title=${column% *}
	#
	unit=${column##* }
	if [[ ${unit} =~ ^temperature.* ]]; then
		unit="[degrees C]"
	fi
	if [[ ${unit} = ${title} ]]; then
		unit=""
	fi
	#
	_title=${title//./_}
	DATA_IDX[${_title}]=${idx}
	DATA_UNIT[${_title}]=${unit}
	((idx++))
done

# for debug
#for _title in "${!DATA_IDX[@]}"; do
#	echo ${DATA_IDX[${_title}]}, ${DATA_UNIT[${_title}]}
#done

# MUST 'index' data in CSV file.
if [[ -z ${DATA_IDX[index]} ]]; then
	puts "Error: Not found 'index' column in CSV file."
	exit 1
fi
# MUST 'timestamp' data in CSV file.
if [[ -z ${DATA_IDX[timestamp]} ]]; then
	puts "Error: Not found 'timestamp' column in CSV file."
	exit 1
fi
# Use 'index' if 'name' does not exist in CSV file.
if [[ -z ${DATA_IDX[name]} ]]; then
	name_idx=${DATA_IDX[index]}
else
	name_idx=${DATA_IDX[name]}
fi

#
# How many GPUs do you have?
#
GPU_COUNT=0
while IFS=, read gpu_id gpu_name; do
	GPUS[${gpu_id}]=${gpu_name# }
	GPU_COUNT=${gpu_id}
done <<<$(awk "BEGIN{FS=\",\"}{ if (\$${DATA_IDX[index]} ~/^[0-9 ]+$/) { print \$${DATA_IDX[index]}\",\"\$${name_idx}}}" ${INPUT_CSV} | sort | uniq)

#
# Split csv file by GPU ID(index).
# Correct data for GNUPLOT. 
#
TMPFILE=$(mktemp)

REGEXP_INDEX="^"
for idx in $(seq 2 ${DATA_IDX[index]}); do
	REGEXP_INDEX="${REGEXP_INDEX}[^,]*, "
done

for gpu_id in $(seq 0 ${GPU_COUNT}); do
	grep -E -e "${REGEXP_INDEX}${gpu_id}," ${INPUT_CSV} \
		| sed -e 's/\[Unknown Error\]//g' \
			  -e 's/\[N\/A\]//g' \
			  -e 's/N\/A//g' \
			  -e 's/ MiB,/,/g' -e 's/ MiB$//' \
			  -e 's/ MHz,/,/g' -e 's/ MHz$//' \
			  -e 's/ W,/,/g' -e 's/ W$//' \
			  -e 's/ %,/,/g' -e 's/ %$//' \
			  > ${TMPFILE}-${gpu_id}
done

#
# Generate GNUPLOT script file.
#
GNUPLOT_SCRIPT=$(mktemp)
for _title in "${!DATA_IDX[@]}"; do
	unit=${DATA_UNIT[${_title}]}
	# Skip if unit is null.
	if [[ -z ${unit} ]]; then
		continue
	fi
	data_idx=${DATA_IDX[${_title}]}
	title=${_title//_/.}
	
	# Check data of ${data_idx}.
	EXEC_PLOT_GPU_COUNT=$(expr ${GPU_COUNT} + 1)
	for gpu_id in $(seq 0 ${GPU_COUNT}); do
		#IS_PLOT_GPU[${gpu_id}]=TRUE
		DATA_VALUE=$(cut -d',' -f${data_idx} ${TMPFILE}-${gpu_id} | sort | uniq)
		if [[ $(wc -l <<<${DATA_VALUE}) -eq 1 ]]; then
			DATA_VALUE=${DATA_VALUE// /}
			if [[ -z ${DATA_VALUE} ]]; then
				#IS_PLOT_GPU[${gpu_id}]=FALSE
				#puts "'${_title}' has no correct data on '${GPUS[${gpu_id}]}'."
				((EXEC_PLOT_GPU_COUNT--))
			fi
		fi
	done

	if [[ ${EXEC_PLOT_GPU_COUNT} -eq 0 ]]; then
		puts "Warning: '${_title}' has no correct data on all GPU. Skipping generating gnuplot script."
		continue
	fi

	{
	puts 'set terminal png'
	puts set output \"${OUT_DIR}/$(basename ${INPUT_CSV%.*})-${title}.png\"
	# x
	puts 'set xlabel ""'
	puts 'set xdata time'
	puts 'set timefmt "%Y/%m/%d %H:%M:%S"'
	puts 'set format x "%H:%M"'
	# y
	puts set ylabel \"${title} ${unit}\"

	puts 'set datafile separator ","'
	puts 'plot \'
	endl=", \\"
	for gpu_id in $(seq 0 ${GPU_COUNT}); do
		if [[ ${gpu_id} -eq ${GPU_COUNT} ]]; then
			endl=""
		fi
		puts \"${TMPFILE}-${gpu_id}\" using ${DATA_IDX[timestamp]}:${data_idx} with lp title "'${GPUS[${gpu_id}]}'"${endl}
	done
	puts ""
	} >> ${GNUPLOT_SCRIPT}
done

#cat ${GNUPLOT_SCRIPT}
gnuplot -c ${GNUPLOT_SCRIPT}

rm -f ${TMPFILE}* ${GNUPLOT_SCRIPT}

exit 0

