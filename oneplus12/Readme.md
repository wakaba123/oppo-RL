## 适配一加12的DQN算法
需要配合info_get使用， info_get的代码在native/jni/temp.cpp里, 编译好的二进制文件在native/libs/arm64-v8a/info_get

```bash
./info_get view #此处只做了单qoe的版本，我选择的是原神的view
```
正常运行info_get后，运行
```python
python DQN_new.py
```

重构了之前的代码的框架，将与设备相关的内容全部抽象到了environment中，网络相关的全部抽象到了DQN_new.py中，方便后续进行网络/设备的切换，两者的交互在于env.reset函数 和 env.step函数。

目前的已知问题
1. power的计算不准确，因为通过time_in_state获得只是获频点所在的时间，但是没有获得当时的利用率，因此精度不准
2. power的计算会在负载重的时候更加不准。
3. 在原神场景下，频点设置会受限制，部分高频点并不能够设置，动作空间实际上会受限。