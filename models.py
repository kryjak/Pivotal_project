import os
import torch as t
from transformers import AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import subprocess
from config import PATH_TO_REPOS, DEVICE
import sys


def clone_deepseek_repo(folder_name: str = "DeepSeek-VL") -> None:
    """
    Clones and installs the DeepSeek-VL repository, and adds it to the Python path.

    Args:
        folder_name (str, optional): The name of the model repository. Defaults to "DeepSeek-VL".

    Returns:
        None

    Note:
        - Clones the repository if it doesn't exist
        - Installs the package in editable mode
        - Adds the repository path to Python's sys.path
        - Returns to the original Pivotal_project directory
    """
    # Assuming your current working directory is 'gitrepos/Pivotal_project'
    path_to_model = os.path.join(PATH_TO_REPOS, folder_name)

    if not os.path.exists(path_to_model):
        print(f"Cloning {folder_name} repo to {path_to_model}")
        os.chdir(PATH_TO_REPOS)
        subprocess.run(
            ["git", "clone", "https://github.com/deepseek-ai/DeepSeek-VL"], check=True
        )
    else:
        print(f"Repo for {folder_name} already exists at {path_to_model}")

    # Change to the DeepSeek-VL directory and install it
    os.chdir(path_to_model)
    subprocess.run(["pip", "install", "-e", "."], check=True)

    # Add DeepSeek-VL to Python path
    if path_to_model not in sys.path:
        sys.path.append(path_to_model)

    # Change back to the Pivotal_project directory
    os.chdir(os.path.join(PATH_TO_REPOS, "Pivotal_project"))

    print(f"DeepSeek-VL installed in editable mode at {path_to_model}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path updated to include: {path_to_model}")


def load_model(model_name):
    """
    Loads and configures a specified vision-language model and its processor.

    Args:
        model_name (str): The name of the model to load. Must be either "DeepSeek-VL", "LLaVa", or "QwenVL".

    Returns:
        tuple: A tuple containing:
            - model: The loaded and configured model
            - processor: The corresponding processor for the model

    Raises:
        ValueError: Implicitly raised if model_name is not one of the supported options.

    Note:
        - For DeepSeek-VL: Uses bfloat16 dtype
        - For LLaVa: Uses 4-bit quantization with float16 dtype
        - For QwenVL: Uses bfloat16 dtype and flash attention 2
    """
    # specify the path to the model
    if model_name == "DeepSeek-VL":
        clone_deepseek_repo()
        from deepseek_vl.models import VLChatProcessor  # type: ignore

        model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(DEVICE)
        model = model.to(t.bfloat16)
        processor = VLChatProcessor.from_pretrained(model_path)

    elif model_name == "LLaVa":
        model_path = "llava-hf/llava-v1.6-vicuna-7b-hf"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=t.float16
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=t.float16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        # model already set to t.float
        processor = LlavaNextProcessor.from_pretrained(model_path)

    elif model_name == "QwenVL":
        model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=t.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        min_pixels = 4 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model, processor
