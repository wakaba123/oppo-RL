@echo off
rem 删除 label_image 文件
del label_image

rem 使用 curl 下载文件
curl -O http://192.168.1.2:8000/bazel-bin/tensorflow/lite/examples/label_image/label_image

rem 使用 adb 将文件推送到安卓设备
adb push label_image /data/local/tmp

rem 修改文件权限
adb shell chmod 777 /data/local/tmp/label_image

echo Script execution completed.
pause
