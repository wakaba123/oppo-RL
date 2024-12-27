..\..\android-ndk-r27c\ndk-build.cmd APP_ABI=arm64-v8a
adb push libs/arm64-v8a/info_get /data/local/tmp
adb shell chmod +x /data/local/tmp/info_get
adb push libs/arm64-v8a/power /data/local/tmp
adb shell chmod +x /data/local/tmp/power