# adb shell su -c "perfetto --txt -c /data/local/tmp/trace_config.txt  -o /data/local/tmp/trace"
# adb shell su -c "chmod 644 /data/local/tmp/trace"
# adb pull /data/local/tmp/trace trace

python record_android_trace -o trace_file.perfetto-trace -t 30s -b 64mb sched freq idle am wm gfx view graphics