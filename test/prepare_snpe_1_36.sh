adb shell rm /data/dlcv/build/ -rf
git clone ssh://git@gitcv.ainirobot.com:10022/inference/model_zoo.git
adb push ./model_zoo/detection/face_ssdlite1_qf_0.35_r2.0/snpe-1.36 /data/dlcv/build/model_zoo/detection/face_ssdlite1_qf_0.35_r2.0/snpe-1.36
adb push ./model_zoo/detection/face_ssdlite1_qf_0.35_r2.0/ssd_fp_debug.png /data/dlcv/build/model_zoo/detection/face_ssdlite1_qf_0.35_r2.0/
wget http://10.60.242.21:8000/release/inference/snpe-1.36/2020-09-01-09-07-36/snpe-1.36.tar.gz
tar xzf snpe-1.36.tar.gz
adb push ./builds/ /data/dlcv/build/
wget http://10.60.242.21:8000/opencv_3.4.8_cpp_shared_ndk17c.zip
unzip opencv_3.4.8_cpp_shared_ndk17c.zip
adb push opencv /data/dlcv/build/
adb push dlcv_test /data/dlcv/build/
adb push libgtest_main.so /data/dlcv/build/
adb push run_snpe_1_36.sh /data/dlcv/build/
adb shell chmod +x /data/dlcv/build/run_snpe_1_36.sh
adb shell /data/dlcv/build/run_snpe_1_36.sh
