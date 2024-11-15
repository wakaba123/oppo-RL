# scp -P 6005 wakaba@118.31.58.129:/home/wakaba/tensorflow/bazel-bin/tensorflow/lite/examples/label_image/label_image .
rm label_image
wget http://192.168.1.2:8000/bazel-bin/tensorflow/lite/examples/label_image/label_image
adb push label_image /data/local/tmp
adb shell chmod 777 /data/local/tmp/label_image