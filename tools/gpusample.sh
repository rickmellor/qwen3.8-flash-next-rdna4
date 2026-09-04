#!/bin/bash
# sample sclk/junction temp/power for the given renderD numbers every 2s until killed; prints CSV
echo "t,gpu,sclk_mhz,tjunc_c,power_w"
while true; do t=$(date +%s); for r in "$@"; do d=$(readlink -f /sys/class/drm/renderD$r/device); h=$(ls -d $d/hwmon/hwmon* | head -1); echo "$t,$(basename $d),$(( $(cat $h/freq1_input)/1000000 )),$(( $(cat $h/temp2_input 2>/dev/null || cat $h/temp1_input)/1000 )),$(( $(cat $h/power1_average 2>/dev/null || cat $h/power1_input)/1000000 ))"; done; sleep 2; done
