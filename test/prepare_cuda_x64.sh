git clone ssh://git@gitcv.ainirobot.com:10022/inference/model_zoo.git
mkdir engine && cd engine
wget http://10.60.242.21:8000/release/inference/openvino/2020-06-04-03-32-03/openvino-2020-02-120.tar.gz
tar xvzf openvino-2020-02-120.tar.gz
wget http://10.60.242.21:8000/release/inference/tensorrt/2020-06-15-11-42-34/tensorrt7.0_cuda9.0.tar.gz
tar xvzf tensorrt7.0_cuda9.0.tar.gz
cd ..
export LD_LIBRARY_PATH=$PWD/engine/builds/inference/openvino/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/engine/builds/inference/openvino/install/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/engine/builds/inference/tensorrt7/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

