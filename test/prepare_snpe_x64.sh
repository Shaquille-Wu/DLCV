git clone ssh://git@gitcv.ainirobot.com:10022/inference/model_zoo.git
mkdir engine && cd engine
wget http://10.60.242.21:8000/release/inference/x86-snpe-1.36/2020-06-16-07-46-24/x86-gcc5.4-snpe-1.36.tar.gz
tar xvzf x86-gcc5.4-snpe-1.36.tar.gz
cd ..
export LD_LIBRARY_PATH=$PWD/engine/builds/inference/snpe/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/engine/builds/inference/snpe/install/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH