adb pull /data/local/tmp/cluster_0.txt .
adb pull /data/local/tmp/cluster_1.txt .
adb pull /data/local/tmp/cluster_2.txt .
adb pull /data/local/tmp/cluster_3.txt .
sed -i '1s/^/freq,time\n/'  cluster_*.txt
python plot_freq.py