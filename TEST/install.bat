python.exe --version
echo requires python 3.12.9+
echo "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe"
python.exe -m pip install --upgrade pip
pip uninstall -y diffusers
rem echo requires CUDA 12.1
rem echo "https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
rem pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision
pip install accelerate optimum-quanto protobuf sentencepiece transformers pandas
pip install git+https://github.com/huggingface/diffusers.git
pause
