import os
import torch as t
from transformers import AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
import subprocess
from config import PATH_TO_REPOS, DEVICE, MODEL

def clone_deepseek_repo(model_id='DeepSeek-VL') -> None:
    path_to_model = os.path.join(PATH_TO_REPOS, model_id)

    if not os.path.exists(path_to_model):
        print(f'Cloning {model_id} repo to {path_to_model}')
        os.chdir(PATH_TO_REPOS)
        subprocess.run(["git", "clone", "https://github.com/deepseek-ai/DeepSeek-VL"], check=True)
        os.chdir(path_to_model)
    else:
        print(f'Repo {model_id} already exists at {path_to_model}')
        os.chdir(path_to_model)
        subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

    print(f'Current working directory: {os.getcwd()}')


def load_model(model_name):
    # specify the path to the model
    if model_name == 'DeepSeek_VL':
        clone_deepseek_repo()
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        
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
