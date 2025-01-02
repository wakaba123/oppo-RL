@echo off

:: Pull files from Android device
adb pull /data/local/tmp/cluster_0.txt .
adb pull /data/local/tmp/cluster_1.txt .
adb pull /data/local/tmp/cluster_2.txt .
adb pull /data/local/tmp/cluster_3.txt .

:: Add header line to each file
for %%i in (cluster_0.txt cluster_1.txt cluster_2.txt cluster_3.txt) do (
    echo freq,time > temp.txt
    type %%i >> temp.txt
    move /y temp.txt %%i > nul
)

:: Run Python script for plotting
python plot_freq.py
