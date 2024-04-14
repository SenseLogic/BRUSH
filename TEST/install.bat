python.exe --version
echo requires python 3.11.7+
echo requires CUDA 12.1
echo "https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
python.exe -m pip install --upgrade pip
pip uninstall accelerate diffusers safetensors torch torchvision transformers xformers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate diffusers pandas safetensors transformers xformers
pause
