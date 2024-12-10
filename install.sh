#!/bin/bash

# Exit script on error
set -e

echo "Creating a virtual environment..."
python3 -m venv llm
source llm/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch..."
# Modify this line based on your CUDA version or use 'cpu' for CPU-only installations.
pip install torch torchvision

echo "Writing requirements.txt..."
cat <<EOT > requirements.txt
deeplake==3.9.27
transformers
numpy
deepspeed
trl
peft
wandb
bitsandbytes
accelerate
evaluate
unsloth
tqdm
neural_compressor
onnx
pandas
scipy
python-dotenv
protobuf
ipykernel
ipywidgets
langchain
langchain-community
langchain-openai
python-docx
openai
EOT

echo "Installing Python packages..."
pip install -r requirements.txt

#echo "Fixing TQDM IProgress warning by enabling ipywidgets..."
pip install --upgrade ipywidgets

#echo "Adding the virtual environment to Jupyter kernels..."
python -m ipykernel install --user --name=llm --display-name "Python (llm)"

#echo "Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y libcufile-dev libaio-dev

echo "Setup complete! Activate your environment using 'source llm/bin/activate'."

