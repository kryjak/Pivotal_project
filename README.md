# Transferability of adversarial attacks between vision-language models

This project explores the transferability of adversarial attacks between different vision-language models (VLMs). It focuses on generating adversarial examples using one VLM and testing their effectiveness on another, with a particular emphasis on jailbreak attacks. The project utilizes models such as DeepSeek-VL and LLaVA, and includes implementations for data preparation, attack generation, and result analysis.

## Installation

To set up the project environment, run the following commands after cloning:
```
cd Pivotal_project
python3.10 -m venv .py310
source .py310/bin/activate
pip install -r requirements.txt
python3.10 -m ipykernel install --user --name=custom_venv --display-name "Custom Venv"
```
You might have to reload your editor window in order for the editor to recognise the new kernel. In VSCode, open the Command Palette (Ctrl+Shift+P) and select "Developer: Reload Window".

You might also want to download the bash script `setup_dev_environment.sh` *before* cloning, run it and follow the instructions. It will:
1) [optional] install a text editor (Vim/Emacs)
2) install Python 3.10
3) [optional] create a new folder for the project
4) clone the project

The terminal commands listed above are printed at the end of the script.

## Environment variables
Before running the code, make sure to set the following environment variables:

- `WANDB_KEY`: for Weights & Biases integration
- `WANDB_ENTITY`: your Weights & Biases username
- `HF_TOKEN`: for Hugging Face model access
- `OPENAI_API_KEY`: for OpenAI API access
- `ANTHROPIC_API_KEY`: for Anthropic API access

In bash, you can set these by adding the following lines to your `.bashrc` file:
```
export KEY_NAME='your_key_here'
```
**NOTE**: Currently, the OpenAI and Anthropic API keys are not used for anything. They will be needed later when we add support for evaluating the jailbreak outputs using such models.


## File Descriptions

- `setup_dev_environment.sh`: Script to set up the development environment
- `main.ipynb`: Main notebook with usage examples
- `prepare_advbench_mini.py`: Script to create a smaller version of the AdvBench dataset for testing
- `VLM_base_classes.py`: Base classes for Vision-Language Models
- `attacks.py`: Implementation of jailbreak attacks on VLMs
- `models.py`: Functions for loading the models
- `config.py`: Configuration settings for the project, including model selection and attack parameters
- `data.py`: Custom DataLoader implementation for VLM jailbreak attacks
- `utils.py`: Utility functions, including WandB integration helpers
- `custom_image_transforms.py`: Custom image transformation functions for adversarial attacks
- `requirements.txt`: List of Python package dependencies

## Usage
All of the code in this repository should be suitable for a single A100 GPU (although you might struggle to train the attacks on an ensemble of contexts while using only 40GB of memory).
To see the code in action, look at the examples in the `main.ipynb` notebook.

**This project is a work in progress and bugs are to be expected.** 
