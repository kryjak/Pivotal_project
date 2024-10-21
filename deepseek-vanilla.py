import torch as t
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
from io import BytesIO
from attrdict import AttrDict
import einops
from torchvision import transforms
from typing import Optional, Union
import os
import base64
import requests


device = 'cuda'

model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
model = model.to(t.bfloat16)
processor = VLChatProcessor.from_pretrained(model_path)

img_size = processor.image_processor.image_size
num_image_tokens = processor.num_image_tokens

model.eval()

for name, param in model.named_parameters():
    param.requires_grad = False

img_size = processor.image_processor.image_size
num_image_tokens = processor.num_image_tokens

tokenizer = processor.tokenizer
processor_mean = t.tensor(processor.image_processor.image_mean).to(device)
processor_std = t.tensor(processor.image_processor.image_std).to(device)

prompt = "What is shown in this image?"
target = "dog"

img_url = "https://wp.inews.co.uk/wp-content/uploads/2023/03/SEI_149780351.jpg?crop=157px%2C0px%2C1537px%2C1537px&resize=640%2C640"
img_path = 'pope_swag.jpg'

def open_image_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the image using PIL
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to retrieve image. Status code: {response.status_code}")
        return None

img = open_image_from_url(img_url)
# img = img.resize((img_size, img_size))
img = img.resize((img_size, img_size))
img.save(img_path)

attack_config_DeepSeek = AttrDict(
    model_name = 'DeepSeek'
    # ,n_epochs = 4
    ,n_epochs = 250
    ,n_logs = 20
    ,eps = 8/255
    ,single_token_target = ["dog"]
    # ,multi_token_target = ['вки', 'deven', '的主要', 'ordin', ' kayaking', '网约车', ' indeed', '崎', '私家', ' startTime'] # random 10-token target
    ,multi_token_target = ['вки', 'deven', '的主要', 'ordin', ' kayaking'] # random 5-token target
    ,optimizer = t.optim.AdamW
    ,lr = 1e-1
    ,weight_decay=1e-2
    ,image_size = model.config.vision_config.params.image_size
)

class DeepSeekVLBaseClass:
    def __init__(self, cfg, model, processor) -> None:
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.device = next(self.model.parameters()).device  # Assuming model is on a specific device

        self.tokenizer = self.processor.tokenizer
        self.embedder = self.model.language_model.get_input_embeddings()

        self.system_prompt = (
                    "You are a helpful language and vision assistant. "
                    "You are able to understand the visual content that the user provides, "
                    "and assist the user with a variety of tasks using natural language."
                )

        self.bos_tokenized = t.tensor([self.tokenizer.bos_token_id]).to(self.device).unsqueeze(0)
        self.system_prompt_tokenized = self.tokenizer.encode(self.system_prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        self.user_tokenized = self.tokenizer.encode('\n\nUser: ', add_special_tokens=False, return_tensors='pt').to(self.device)
        self.assistant_tokenized = self.tokenizer.encode('\n\nAssistant:', add_special_tokens=False, return_tensors='pt').to(self.device)

        # Set up embeddings
        self.bos_embedded = self.embedder(self.bos_tokenized)
        self.system_prompt_embedded = self.embedder(self.system_prompt_tokenized)
        self.user_embedded = self.embedder(self.user_tokenized)
        self.assistant_embedded = self.embedder(self.assistant_tokenized)

    # Define model-specific methods
    def generate_autoregressive_with_pil(self, prompt: str, image: Image.Image, max_new_tokens: Optional[int] = 1, no_eos_token: Optional[bool] = False, use_cache: Optional[bool] = False, **kwargs):
        if isinstance(image, Image.Image):
           buffered = BytesIO()
           # Important that this is PNG or another lossless compression type!
           # I originally used JPEG, but this rescaled all images to the [0, 255]
           # range after loading
           image.save(buffered, format="PNG")
           image_str = base64.b64encode(buffered.getvalue()).decode()
           image_uri = f"data:image/PNG;base64,{image_str}"

           conversation = [
               {
                   "role": "User",
                   "content": f"<image_placeholder>{prompt}",
                   "images": [image_uri]
               },
               {
                   "role": "Assistant",
                   "content": ""
               }
           ]

           # we had a PIL image to begin with, but conversation only accepts paths or base64
           pil_image = load_pil_images(conversation)

           prepare_inputs = self.processor(
               conversations=conversation,
               images=pil_image,
               force_batchify=True
               ).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        if no_eos_token:
            stopping_criterion = None
        else:
            stopping_criterion = self.tokenizer.eos_token_id

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=stopping_criterion,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache, # does caching here matter?
            output_logits=True,
            return_dict_in_generate=True,
            **kwargs
        )

        completion = self.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=False)
        return outputs, completion

    def generate_autoregressive(self, prompt: str, image: t.Tensor, system_prompt: Optional[str] = None, max_new_tokens: Optional[int] = 1, no_eos_token: Optional[bool] = False, use_cache: Optional[bool] = False, **kwargs):
        inputs_embeds = self.prepare_inputs_grad(prompt, image, system_prompt)
        attention_mask = t.ones((1, inputs_embeds.shape[1]), dtype=t.long).to(self.device)

        if no_eos_token:
            stopping_criterion = None
        else:
            stopping_criterion = self.tokenizer.eos_token_id

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            # attention_mask=prepare_inputs.attention_mask,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=stopping_criterion,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache, # does caching here matter?
            output_logits=True,
            return_dict_in_generate=True,
            **kwargs
        )

        completion = self.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=False)
        return outputs, completion

    def preprocess_image(self, image: t.Tensor):
        img_size = model.config.vision_config.params.image_size
        prep = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Resize(size=(img_size, img_size), interpolation=transforms.functional.InterpolationMode.BICUBIC, antialias=True),
            # transforms.Normalize(mean=processor_mean, std=processor_std)
        ])

        return prep(image)

    def embed_image(self, tensor_image: t.Tensor):
        assert isinstance(tensor_image, t.Tensor)

        if tensor_image.ndim != 5:
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

        n_batches, n_images = tensor_image.shape[0:2]
        images = einops.rearrange(tensor_image, "b n c h w -> (b n) c h w")
        image_embedded = self.model.aligner(self.model.vision_model(images))
        image_embedded = einops.rearrange(image_embedded, "(b n) t d -> b (n t) d", b=n_batches, n=n_images)

        return image_embedded

    def prepare_inputs_grad(self, prompt: str, image: Optional[t.Tensor] = None, past_output: Optional[str] = None, system_prompt: Optional[str] = None):
        prompt_tokenized = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        prompt_embedded = self.embedder(prompt_tokenized)

        if system_prompt != None:
            system_prompt_tokenized = self.tokenizer.encode(system_prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            system_prompt_embedded = self.embedder(system_prompt_tokenized)
        else:
            system_prompt_embedded = self.system_prompt_embedded

        inputs_embedded = t.cat((self.bos_embedded,
                                system_prompt_embedded,
                                self.user_embedded),
                                dim=1).to(self.device)

        # insert image embeddings if an image is provided
        if image != None:
            image_embedded = self.embed_image(self.preprocess_image(image))
            inputs_embedded = t.cat((inputs_embedded, image_embedded), dim=1)

        inputs_embedded = t.cat((inputs_embedded,
                                prompt_embedded,
                                self.assistant_embedded),
                                dim=1).to(self.device)

        # append output of previous generations - used for manual autoregressive loops
        if bool(past_output): # bool('') == False
            past_output_embedded = self.embedder(self.tokenizer.encode(past_output, add_special_tokens=False, return_tensors='pt').to(self.device))
            inputs_embedded = t.cat((inputs_embedded, past_output_embedded), dim=1)

        return inputs_embedded

    def generate_token_grad(self, prompt: str, image: Optional[t.Tensor] = None, use_cache: Optional[bool] = False, past_key_values = None, past_output: Optional[str] = None, **kwargs):
        inputs_embeds = self.prepare_inputs_grad(prompt, image, past_output)
        # not sure if this if statement is necessary
        # is past_key_values=None fine?
        if use_cache:
            output = self.model.language_model.forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                **kwargs
                )
        else:
            output = self.model.language_model.forward(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                **kwargs
                )

        return output

    def generate_autoregressive_manual(self, prompt: str, image: Optional[t.Tensor] = None, use_cache: Optional[bool] = False, max_new_tokens: Optional[int] = 1, no_eos_token: Optional[bool] = False, **kwargs):
        generated_tokens = []
        accumulated_final_logits = t.tensor([]).to(self.device)
        past_key_values = None
        past_output = ''

        with t.no_grad():
            for step in range(max_new_tokens):
                output = self.generate_token_grad(prompt, image, use_cache=use_cache, past_key_values=past_key_values, past_output=past_output, **kwargs)
                past_key_values=output.past_key_values

                new_token_logits = output.logits[:, -1]
                accumulated_final_logits = t.cat((accumulated_final_logits, new_token_logits))

                new_token = tokenizer.decode(new_token_logits.argmax(-1))
                # if the new token is <end_of_sentence>, then we either 1) break, or 2) pick the second most likely token and continue generating
                if new_token == self.tokenizer.eos_token:
                    if not no_eos_token:
                        generated_tokens.append(new_token)
                        break
                    else:
                        second_largest_token = new_token_logits.topk(k=2, dim=-1, largest=True).indices[:, -1]
                        new_token = tokenizer.decode(second_largest_token)

                generated_tokens.append(new_token)

                past_output += new_token

        accumulated_final_logits = einops.rearrange(accumulated_final_logits, '(new_tokens batch) d_vocab -> new_tokens batch d_vocab', new_tokens=1)

        return accumulated_final_logits, generated_tokens
    
base_model_class = DeepSeekVLBaseClass

class ControlSingleTokenAttack(base_model_class):
    def __init__(self, cfg, model, procesor, wandb_name = None) -> None:
        super().__init__(cfg, model, processor)
        # if self.cfg.wandb_name == None:
        #     self.cfg.wandb_name = wandb_name

        #wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_name, config=self.cfg)
        #run_path = wandb.run.path
        #print(f'Current run path: {run_path}')

    def train_attack(self,
                     prompt: Union[str, list[str]],
                     image: Union[Image.Image, list[Image.Image]],
                     target: Optional[str] = None,
                     verbose: Optional[bool] = True) -> tuple[Image.Image, t.Tensor, list[float]]:

        if target is None:
            target = self.cfg.single_token_target[0]

        init_image = transforms.ToTensor()(image).to(t.bfloat16).to(self.device)
        delta = t.zeros_like(init_image, dtype=t.bfloat16, requires_grad=True, device=self.device)

        target_tokenized = self.tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').to(self.device)

        # we will optimise the perturbation mask, not the original image
        optimizer = self.cfg.optimizer([delta], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        eps = self.cfg.eps

        loss_train = []

        for step in range(self.cfg.n_epochs):
            optimizer.zero_grad()

            # should I normalise the perturbed image?
            perturbed_init_image = (init_image + eps * delta.clamp(-1, 1)).clamp(0, 1).to(self.device) * 255.0 # this will be a t.Tensor, not a PIL.Image.Image, but the range is [0.0, 255.0]

            output = self.generate_token_grad(prompt, perturbed_init_image)
            logits = output.logits # [batch, sequence_position, vocab]
            next_token_logits = logits[:, -1] # [batch, vocab]
            next_token_pred = tokenizer.batch_decode(next_token_logits.argmax(-1), skip_special_tokens=True)

            loss_fn = t.nn.CrossEntropyLoss()
            loss = loss_fn(next_token_logits, target_tokenized[:, -1])
            loss.backward()
            loss_train.append(loss.item())
            #wandb.log({"train_loss": loss.item()}, step=step+1)

            optimizer.step()

            if (step+1) % (self.cfg.n_epochs // self.cfg.n_logs) == 0:
                next_token_probs = t.softmax(next_token_logits, dim=-1)
                target_token_prob = next_token_probs[0, target_tokenized.item()]
                #wandb.log({'next_token_pred': next_token_pred, 'target_token_prob': target_token_prob.item()}, step=step+1)
                if verbose:
                    print(f'Step {step+1}:')
                    print(f'loss: {loss.item():.4f}')
                    print(f'next_token_pred: {next_token_pred}')
                    print(f'target_token_prob: {target_token_prob.item():.4f}')
                    print('------------------')

            t.cuda.empty_cache()

        # path_to_tensors = '/content/sample_data/'
        # for tensor_name in ['init_image', 'delta']:
        #     t.save(eval(tensor_name), path_to_tensors + tensor_name + '.pt')
        #     #wandb.save(path_to_tensors + tensor_name + '.pt', base_path='/content')

        return init_image, delta, loss_train

    def execute_attack(self,
                        prompt: Union[str, list[str]],
                        image: Optional[t.Tensor] = None,
                        delta: Optional[t.Tensor] = None,
                        adversarial_image: Optional[t.Tensor] = None,
                        generation_method: Optional[str] = 'automatic',
                        max_new_tokens: Optional[int] = 1,
                        no_eos_token: Optional[bool] = False,
                        **kwargs
                       ) -> list[str]:

        # should I test autoregressively or still using Stan's method?
        # I guess autoregressively, as we should use as little of the setup from
        # the training phase as possible
        # when transferring this attack to a black-box model, we won't be able to
        # use anything else than the offical API
        if adversarial_image == None:
            adversarial_image = (image + self.cfg.eps*delta.clamp(-1, 1)).clamp(0, 1)
            # adversarial_image = transforms.ToPILImage()(adversarial_image.float())
            adversarial_image = adversarial_image * 255.0

        with t.no_grad():
            if generation_method == 'automatic':
                output, answer = self.generate_autoregressive(prompt, adversarial_image, max_new_tokens=max_new_tokens, no_eos_token=no_eos_token, **kwargs)
            elif generation_method == 'manual':
                output, answer = self.generate_autoregressive_manual(prompt, adversarial_image, max_new_tokens=max_new_tokens, no_eos_token=no_eos_token, **kwargs)
            else:
                raise NotImplementedError(f'Generation method {generation_method} not implemented yet.')

        #wandb.log({'answer': answer})

        return output, answer

base_instance = base_model_class(attack_config_DeepSeek, model, processor)
single_token_attack = ControlSingleTokenAttack(attack_config_DeepSeek, model, processor)
init_image, delta, loss_train = single_token_attack.train_attack(prompt, img)
