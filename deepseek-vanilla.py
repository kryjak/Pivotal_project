import torch as t
from models import load_model
from attacks import ControlSingleTokenAttack
from config import *
from utils import *
from VLM_base_classes import DeepSeekVLBaseClass

model, processor = load_model(MODEL)
attack_config = attack_config_DeepSeek + wandb_config
device = DEVICE

img_size = processor.image_processor.image_size

model.eval()

for name, param in model.named_parameters():
    param.requires_grad = False

# THIS NEEDS TO BE DONE AFTER SWITCHING THE GRADIENTS OFF
base_model_class = DeepSeekVLBaseClass
base_instance = base_model_class(model, processor)

tokenizer = processor.tokenizer
processor_mean = t.tensor(processor.image_processor.image_mean).to(device)
processor_std = t.tensor(processor.image_processor.image_std).to(device)

prompt = "What is shown in this image?"

img_url = "https://wp.inews.co.uk/wp-content/uploads/2023/03/SEI_149780351.jpg?crop=157px%2C0px%2C1537px%2C1537px&resize=640%2C640"
img_path = 'pope_swag.jpg'

img = open_image_from_url(img_url)
img = img.resize((img_size, img_size))
img.save(img_path)

single_token_attack = ControlSingleTokenAttack(base_instance, attack_config, wandb_name='single_token')
init_image, delta, loss_train = single_token_attack.train_attack(prompt, img)
