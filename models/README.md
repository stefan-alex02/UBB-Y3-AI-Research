
## Setup instructions
When creating a new .venv for the project, make sure to install the following dependencies using the following command (do not install pytorch before running this command):

- For CUDA 12.x:

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

- For CUDA 11.8:

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118