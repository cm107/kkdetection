# kkdetection
- This package has tools for some "object detection" packages.
- Support for detectron2( https://github.com/facebookresearch/detectron2 ).
- Support for bytetrack( https://arxiv.org/abs/2110.06864 )
- Support for PaddleDetection( https://github.com/PaddlePaddle/PaddleDetection )

## Installation
1. Pytorch

First, You will need to manually install pytorch for your machine environment (see: https://pytorch.org/get-started/locally/ )!!

ex) pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

2. PaddlePaddle

If you want to use PaddleDetection, you need installing PaddlePaddle( see: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html ).

ex)
```
python -m pip install paddlepaddle-gpu==2.2.2.post111 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# If cpu, python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple
pip install protobuf==3.20.1
python -c "import paddle; paddle.utils.run_check()"
```

```
(kkdetection) root@593514f23031:~# python -c "import paddle; paddle.utils.run_check()"
Running verify PaddlePaddle program ... 
W1114 22:57:32.420543    32 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 11.2, Runtime API Version: 11.1
W1114 22:57:32.436527    32 device_context.cc:465] device: 0, cuDNN Version: 8.0.
W1114 22:57:33.949204    32 device_context.h:397] WARNING: device: 0. The installed Paddle is compiled with CUDNN 8.1, but CUDNN version in your machine is 8.0, which may cause serious incompatible bug. Please recompile or reinstall Paddle with compatible CUDNN version.
PaddlePaddle works well on 1 GPU.
PaddlePaddle works well on 1 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

3. Cython

```
pip install Cython
```

3. kkdetection

```
pip install 'git+https://github.com/kazukingh01/kkdetectron2.git'
```
