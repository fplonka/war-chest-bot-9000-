#!/usr/bin/env bash
# Block until both cards are actually free.
#
# A killed probe does not give its memory back the moment the signal lands:
# the driver reclaims a solve farm's sixteen gigabytes over seconds. Measuring
# before that finishes reads as an out-of-memory failure of whatever ran next,
# and it has cost three measurements so far.
want=${1:-2000}
for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${busy:-99999}" -lt "$want" ] && { echo "cards free (${busy} MiB)"; exit 0; }
    sleep 5
done
echo "cards still hold ${busy} MiB after five minutes" >&2
exit 1
