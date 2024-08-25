python.exe --version
echo requires python 3.11.7+
echo requires CUDA 12.1
echo "https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
python.exe -m pip install --upgrade pip
pip uninstall -y diffusers
pip install accelerate optimum-quanto protobuf sentencepiece torch torchvision transformers
pip install git+https://github.com/huggingface/diffusers.git
pause
