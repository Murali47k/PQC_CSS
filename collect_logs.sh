#!/bin/bash

OUTPUT="docker_logs.txt"

> "$OUTPUT"

for i in {0..9}; do
    echo "==================== CONTAINER $i ====================" >> "$OUTPUT"
    docker logs "container$i" >> "$OUTPUT" 2>&1
    echo -e "\n\n" >> "$OUTPUT"
done

echo "Logs saved to $OUTPUT"

# for i in {0..9}; do
#     echo "========== container$i =========="
#     docker logs container$i 2>&1 | grep -A10 -B2 "Round 11"
# done