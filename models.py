import os
import torch as t
from transformers import AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
import subprocess
from config import PATH_TO_REPOS, DEVICE, MODEL
import sys

def clone_deepseek_repo(model_id='DeepSeek-VL') -> None:
    # Assuming your current working directory is 'gitrepos/Pivotal_project'
    path_to_model = os.path.join(PATH_TO_REPOS, model_id)

    if not os.path.exists(path_to_model):
        print(f'Cloning {model_id} repo to {path_to_model}')
        os.chdir(PATH_TO_REPOS)
        subprocess.run(["git", "clone", "https://github.com/deepseek-ai/DeepSeek-VL"], check=True)
    else:
        print(f'Repo for {model_id} already exists at {path_to_model}')

    # Change to the DeepSeek-VL directory and install it
    os.chdir(path_to_model)
    subprocess.run(["pip", "install", "-e", "."], check=True)

    # Add DeepSeek-VL to Python path
    if path_to_model not in sys.path:
        sys.path.append(path_to_model)

    # Change back to the Pivotal_project directory
    os.chdir(os.path.join(PATH_TO_REPOS, 'Pivotal_project'))

    print(f'DeepSeek-VL installed in editable mode at {path_to_model}')
    print(f'Current working directory: {os.getcwd()}')
    print(f'Python path updated to include: {path_to_model}')


def load_model(model_name):
    # specify the path to the model
    if model_name == 'DeepSeek-VL':
        clone_deepseek_repo()
        from deepseek_vl.models import VLChatProcessor
        
        model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True).to(DEVICE)
        model = model.to(t.bfloat16)
        processor = VLChatProcessor.from_pretrained(model_path)

    elif model_name == 'LLaVa':
        model_path = "llava-hf/llava-v1.6-vicuna-7b-hf"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=t.float16)

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=t.float16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config)
        # model already set to t.float
        processor = LlavaNextProcessor.from_pretrained(model_path)

    return model, processor
