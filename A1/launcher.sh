#!/bin/bash

dir=log_dir

otypes=("O0" "O1" "O2" "O3")
mat_size=11
types=("SIMPLE" "BLOCK")

if [ ! -d $dir ]; then
    echo "$dir not existant. Creating it.";
    mkdir $dir
else
    echo "$dir exists. Doing nothing.";
fi

if [ ! -d "$dir/runs" ]; then
    echo "$dir/runs not existant. Creating it.";
    mkdir $dir/runs
fi

calculate_num_repeats() {
    local num_repeats=$(echo "$2 / 4" | bc)
    num_repeats=$(printf "%.0f" "$num_repeats")

    local target_time=1.5  # Target execution time in seconds
    local max_execution_time=0

    # Measure the maximum execution time of transpose
    for ((i=0; i<5; i++)); do
        local execution_time=$( { /usr/bin/time -f "%e" ./bin/transpose $1 $num_repeats; } 2>&1 >/dev/null )
        if (( $(echo "$execution_time > $max_execution_time" | bc -l) )); then
            max_execution_time=$execution_time
        fi
    done

    # Increase num_repeats exponentially until the maximum execution time exceeds the target time
    while (( $(echo "$max_execution_time < $target_time" | bc -l) )); do
        num_repeats=$((num_repeats * 2))
        max_execution_time=$( { /usr/bin/time -f "%e" ./bin/transpose $1 $num_repeats; } 2>&1 >/dev/null )
    done

    echo "$num_repeats"
}

get_value() {
    local optimization="$1"
    local type="$2"
    local number="$3"

    local value

    # Find the line containing the value and print only the first match
    value=$(awk -v opt="$optimization" -v t="$type" -v num="$number" '$0 ~ (opt " - " t " - " num) {print $NF; exit}' log.txt)

    value=$(echo "$value / 2" | bc)
    value=$(printf "%.0f" "$value")

    echo "$value"
}

num_repeats=2000

for otype in ${otypes[@]}; do
    for type in ${types[@]}; do
        make clean

        if [ "$type" = "SIMPLE" ]; then
            make $otype LOG=LOG
        else
            make $otype LOG=LOG BLOCK=BLOCK
        fi

        for ((i=1; i<mat_size; i++)); do
            echo "$otype - $type - $i"

            # num_repeats=$(calculate_num_repeats $i $num_repeats)
            num_repeats=$(get_value $otype $type $i)

            echo $num_repeats
            valgrind --tool=cachegrind --cache-sim=yes ./bin/transpose $i &> "$dir/${otype}_${type}_${i}_valgrind.log"
            

            ./bin/transpose $i $num_repeats >> "$dir/runs/${otype}_${type}_${i}.csv"
        done
    done
done
