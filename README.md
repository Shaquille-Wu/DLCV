# **DLCV**  
###### Pipeline for Deep Learning based Computer Vision algorithms
ONE json define a DL algorithm

---

### ConfigureFile Format
```json
{
      "preprocess":{
          "debug": true,
          "ops":[
              {"type":"convert_data_type", "param":{"dtype": "float32"}},
              {"type":"resize",      "param":{"width":320, "height":320, "mode":"bilinear"}},
              {"type":"normalize",   "param":{"mean": 127.5, "std": 127.5}},
              {"type":"totensor",    "param":{"hwc2chw": true, "swapchannel":true, "dtype":"float32"}}
          ]
      },
      "inference": {
          "debug": true,
          "engine_param":{
              "thread_num": 1,
              "gpu_id": 0
          },
          "engine": "libncnn.so",
          "model":  "ncnn/face_detect",
          "inputs":  [{"name":"data",          "shape":[1, 3, 320, 320], "dtype":"float32"}],
          "outputs": [{"name":"detection_out", "shape":[1, 1, 100, 6],   "dtype":"float32"}]
      },
      "postprocess":{
          "debug": true,
          "ops":[
              {
                  "type":"det_ssd_post", 
                  "param":{
                       "conf_thresh": 0.4,
                       "label_map":{"face":1, "body":2}
                  }
              }
          ]
      }
}

```

  
#### Interface: 
| Name               | Description                     |
| --                 | :---                            |
|Detector            | 检测器                          |
|Multiclass          | 分类/多属性                     |
|Feature             | reid特征，如人脸识别，人体reid  |
|FeaturemapRunner    | 提取featuremap，如分割/eco_featuremap|   
|KeypointsAttributes | 关键点with属性融合模型,如角度   |
|IModelRunner        | 更通用的接口，可替代上面所有接口|

#### Preprocess:
| Name               | Description                     |
| --                 | :---                            |
|convert_data_type   | 转换数据类型，如float32，int8   |
|resize              | 指定shape的resize               |
|meanfile            | 均值文件                        |
|normalize           | 归一化：y = (x - mean)/std      |
|batchnorm           ||                                   
|swapchannel         | 交换color通道，rgb<->bgr        |
|togray              | 转为灰度图                      |
|graytobgr           | 灰度转bgr，单通道复制为三通道   |
|totensor            | cv::Mat copy到tensor buffer     |

#### Inference:
支持IInference接口的inference engine均可,目前有snpe，tensorrt，openvino，ncnn

#### Postprocess:
| Name                       | Description                     |
| --                         | :---                            |
|det_ssd_post                | ssd检测的后处理                 |
|det_ssd_detectionout_post   | ssd检测的后处理, 带自定义实现的detectionout|
|featuremap_post             | 用于提取featuremap的模型，如分割/eco_featuremap等 |
|reidfeature_post            | reidfeature后处理，用于reid模型，如人脸识别/人体reid等|
|multiclass_post             | 用于分类/多属性|
|kps_attribute_post          | 用于关键点+分类/多属性，融合模型|


* kps_attribute_post example:

```json
    {
        "type":"kps_attribute_post", 
        "param":{
            "keypoints_type": "heat_map",  # or regress
            "data_format": "nchw",
            "keypoints_idx_range": [0, 105],
            "keypoints_orders":[
                {
                    "name":"outline", "out_idx_range": [0, 32],
                    "reorder": [0, 4, 6, 2, ..., 46]  # 如果不为空，就做reorder
                },
                {
                    "name":"eye", "out_idx_range": [33, 52],
                    "reorder": [52, 53, 72, 54, 56, 57, 73, 74, 55, 104, 58, 59, 60, 61, 62, 63, 75, 76,105, 77]
                },
                {
                    "name":"eyebrow", "out_idx_range": [53, 70],
                    "reorder": [...]
                },
                {
                    "name":"nose", "out_idx_range": [71, 85],
                    "reorder": [...]
                },
                {
                    "name":"mouth", "out_idx_range": [86, 105],
                    "reorder": [...]
                },
            ]
        }
    }
```

---
### HowTo
####  Build dlcv library on  Linux / Android / iOS

#####  build for x86 Linux
install opencv
```shell
sudo apt-get install libopencv-dev
```

* build for x86_64 Linux using gcc
```shell
cd <dlcv-root-dir>
mkdir build-linux-x86_64-gcc && cd build-linux-x86_64-gcc
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/x86_64-linux.gcc.toolchain.cmake
make -j8 
```

* build for x86_64 Linux using clang
```shell
cd <dlcv-root-dir>
mkdir build-linux-x86_64-clang && cd build-linux-x86_64-clang
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/x86_64-linux.clang.toolchain.cmake
make -j8 
```

#####  build for Android
get android-ndk
```shell
download android-ndk from http://developer.android.com/ndk/downloads/index.html
$ unzip android-ndk-r16b-linux-x86_64.zip
$ export NDK_HOME=<your-ndk-root-path>
```

(optional) drop debug compile flag to reduce binary size due to [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)
```shell
# edit $ANDROID_NDK/build/cmake/android.toolchain.cmake with your favorite editor
# remove "-g" line
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

get opencv for android sdk
```shell
# run the scripts in `deps/` to download prebuilt opencv library or using any version you like 
$ cd <dlcv-root-dir>/deps
$ ./download_opencv_ndk17c_clang6.sh
```

* build for Android armv8
```shell
cd <dlcv-root-dir>
mkdir build-android-armv8 && cd build-android-armv8
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_STL=c++_shared \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-27
make -j8
```

* build for Android armv7
```shell
cd <dlcv-root-dir>
mkdir build-android-armv7 && cd build-android-armv7
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_STL=c++_shared \
    -DANDROID_ABI="armeabi-v7a" \
    -DANDROID_PLATFORM=android-22
make -j8
```

* `ANDROID_STL` could be `gnustl_shared/gnustl_static` or `c++_shared/c++_static`, [more detail](https://developer.android.com/ndk/guides/cpp-support)

---
### features
- debug模式, json中开启debug，可存图
- 模型精度评测工具

### todo
- [ ] 模型全覆盖单元测试
- [ ] add profiler 性能评估
- [ ] batchsize > 1
- [ ] fastrcnn/ rfcn 多输入问题

### 设计原则
* 关于异常处理：
    * 初始化阶段异常，就地崩溃，打出错误原因，并提示修复方式
    * 运行时阶段异常（如数据异常），逐层返回错误码，不能崩溃
* 配置文件设计（json）
    * 关键字意义明确：「易懂」的基础上考虑「简洁」
    * 尽量少用默认参数：让参数显式配置，让用户经常看到，助于理解框架具备的功能，而不是去猜有哪些隐藏功能


## Release notes
### 2020.4.25  V0.1
  Base Release 
  support xxx
  



