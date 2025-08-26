echo -1 > /proc/sys/kernel/perf_event_paranoid
echo off | sudo tee /sys/devices/system/cpu/smt/control
for i in $(seq 0 7);
do
    echo 400000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
    echo 4500000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
    echo powersave > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
    echo balance_performance > /sys/devices/system/cpu/cpufreq/policy$i/energy_performance_preference
    #echo 4500000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_min_freq
    #echo 4500000 > /sys/devices/system/cpu/cpufreq/policy$i/scaling_max_freq
    #echo performance > /sys/devices/system/cpu/cpufreq/policy$i/scaling_governor
    #echo performance > /sys/devices/system/cpu/cpufreq/policy$i/energy_performance_preference
done
