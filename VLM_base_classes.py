from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import torch as t
from config import DEVICE
from transformers.models.llava_next.modeling_llava_next import unpad_image, unpad_image
from io import BytesIO
import base64
from typing import Optional
import einops


class LlavaBaseClass:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor
        self.device = DEVICE

        self.tokenizer = self.processor.tokenizer
        self.embedder = self.model.get_input_embeddings()

        self.system_prompt = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
        )

        self.bos_tokenized = t.tensor([self.tokenizer.bos_token_id]).to(self.device).unsqueeze(1)
        self.newline_tokenized = self.tokenizer.encode('\n', add_special_tokens=False, return_tensors='pt').to(self.device)
        self.system_prompt_tokenized = self.tokenizer.encode(self.system_prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        self.user_tokenized = self.tokenizer.encode(' USER: ', add_special_tokens=False, return_tensors='pt').to(self.device)
        # tokenized prompt goes here - it will be defined later on
        self.assistant_tokenized = self.tokenizer.encode(' ASSISTANT:', add_special_tokens=False, return_tensors='pt').to(self.device)

        # Set up embeddings
        self.bos_embedded = self.embedder(self.bos_tokenized)
        self.newline_embedded = self.embedder(self.newline_tokenized)
        self.system_prompt_embedded = self.embedder(self.system_prompt_tokenized)
        self.user_embedded = self.embedder(self.user_tokenized)
        # embedded prompt goes here - it will be defined later on
        self.assistant_embedded = self.embedder(self.assistant_tokenized)

    # Define model-specific methods
    def generate_autoregressive(self, prompt: str, image: Image.Image, max_new_tokens: int, no_eos_token: Optional[bool] = False):
        assert isinstance(image, Image.Image), "Image must be a PIL image"

        # NORMAL LLAVA PIPELINE
        # direct_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
        # prepare_inputs = self.processor(
        #             direct_prompt,
        #             image,
        #             return_tensors="pt",
        #             padding=True,
        #             do_pad=True).to(self.device)

        # inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # output = self.model.forward(
            # **prepare_inputs,
            # return_dict=True,
            # output_hidden_states=False,
            # output_attentions=False,
            # use_cache=False,
        # )

        # OUR OWN IMPLEMENTATION
        inputs_embeds = self.prepare_inputs_grad(prompt, transforms.PILToTensor()(image).float())

        # training a single token attack might cause the second generated token
        # to be EoS. So we add an option to generate tokens beyond that.
        if no_eos_token:
            stopping_criterion = None
        else:
            stopping_criterion = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=t.ones(inputs_embeds.shape[1]).unsqueeze(0),
            position_ids=t.arange(inputs_embeds.shape[1], dtype=t.long, device=self.device).unsqueeze(0),
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=stopping_criterion,
            max_new_tokens=max_new_tokens,
            # past_key_values=None, ???
            output_hidden_states=False,
            output_attentions=False,
            use_cache=True,
            return_dict_in_generate=True,
            do_sample=False, # these two are needed for greedy sampling
            num_beams=1, # but they are the default settings anyway
            )

        completion = self.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
        return outputs, completion

    def preprocess_image(self, image: t.Tensor, processor_mean: Optional[t.Tensor] = None, processor_std: Optional[t.Tensor] = None) -> t.Tensor:
        """
        Preprocesses the image tensor to the required format.
        """
        if processor_mean is None:
            processor_mean = t.tensor(self.processor.image_processor.image_mean).to(self.device)
        if processor_std is None:
            processor_std = t.tensor(self.processor.image_processor.image_std).to(self.device)

        img_size = self.model.config.vision_config.image_size

        prep = transforms.Compose([
            transforms.Lambda(lambda x: x/255.0),
            transforms.Resize(size=(img_size, img_size), interpolation=F.InterpolationMode.BICUBIC, antialias=True), # resample=3 corresponds to BICUBIC
            transforms.Normalize(mean=processor_mean, std=processor_std)
        ])

        return prep(image)

    def embed_image(self, image: t.Tensor) -> t.Tensor:
        """
        Embeds the image tensor into the embedding space.
        """
        assert isinstance(image, t.Tensor) and image.ndim == 3, "Tensor image must be 3D"
        # we will only work with images <336x336, so the number of patches is 3
        # this needs to be supplied as a list
        image_num_patches = [3]

        image_features = self.model.vision_tower(image, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.model.config.vision_feature_layer] # model.config.vision_feature_layer == -2
        selected_image_feature = selected_image_feature[:, 1:]

        # aligns embedded image with the embedding space of the text
        image_embedded = self.model.multi_modal_projector(selected_image_feature)
        image_embedded = t.split(image_embedded, image_num_patches, dim=0)

        image_sizes = t.tensor([[image.size(-2), image.size(-1)]], device=self.device)
        # removes padding and appends an <image_newline> token at the end of each row
        image_embedded, _ = self._pack_image_features(
            image_embedded,
            image_sizes,
            image_newline=self.model.image_newline
        )

        # unsqueeze to pretend we have batch size
        image_embedded = image_embedded.unsqueeze(0)

        return image_embedded

    def prepare_inputs_grad(self, prompt: str, image: t.Tensor):
        prompt_tokenized = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        prompt_embedded = self.embedder(prompt_tokenized)

        image_patches = self.get_and_join_patches(image)
        image_preprocessed = self.preprocess_image(image_patches)
        image_embedded = self.embed_image(image_preprocessed)

        inputs_embedded = t.cat((self.bos_embedded,
                            self.system_prompt_embedded,
                            self.user_embedded,
                            image_embedded,
                            prompt_embedded,
                            self.newline_embedded,
                            self.assistant_embedded),
                            dim=1).to(self.device)

        return inputs_embedded

    def generate_token_grad(self, prompt: str, image: t.Tensor, **kwargs):
        inputs_embeds = self.prepare_inputs_grad(prompt, image)
        output = self.model.forward(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            **kwargs)

        return output

    def _pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        After embedding, LlaVa removes features corresponding to the paddings. They also append a special <image_newline> token at the end of each row of features to provide an explicit indication of the shape of the image.
        Copied (and slighly simplified) from:
        https://github.com/huggingface/transformers/blob/86a1269e19af022e04bc2aad82572cd5a9e8cdd9/src/transformers/models/llava_next/modeling_llava_next.py#L799
        """

        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size # 336//14 = 24

                num_patch_width, num_patch_height = (1, 2)

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])

                if image_newline is not None:
                    image_feature = t.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = t.cat((base_image_feature, image_feature), dim=0)

            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))

        all_image_features = t.cat(new_image_features, dim=0)
        feature_lens = t.tensor(feature_lens, dtype=t.long, device=self.device)

        return all_image_features, feature_lens


    def get_and_join_patches(self, tensor_image):
        """WARNING: this assumes the image is < 336x336px, such that it will be processed into 3 patches by LlaVa-Next"""

        # split the image into two halves and pad with 0s (black background)
        left_half, right_half = t.split(tensor_image, tensor_image.shape[-1]//2, -1)

        padding_left_half = t.zeros_like(left_half)
        new_patch_left = t.cat((padding_left_half, left_half), dim=-1)

        padding_right_half = t.zeros_like(right_half)
        new_patch_right = t.cat((right_half, padding_right_half), dim=-1)

        # stack the patches together
        # this assumes there are 3 patches, so that they are arranged in a 1x3 grid
        # for more patches, this might be e.g. 2x2
        # see https://llava-vl.github.io/blog/2024-01-30-llava-next/
        input_tensor = t.stack((tensor_image, new_patch_left, new_patch_right), dim=0)

        return input_tensor


class DeepSeekVLBaseClass:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor
        self.device = DEVICE

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

        try:
            from deepseek_vl.utils.io import load_pil_images
            self.load_pil_images = load_pil_images
        except ImportError as e:
            print(f"Warning: Could not import DeepSeek utilities: {e}")

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
           pil_image = self.load_pil_images(conversation)

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
        img_size = self.model.config.vision_config.params.image_size
        prep = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Resize(size=(img_size, img_size), interpolation=F.InterpolationMode.BICUBIC, antialias=True),
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

    def generate_autoregressive_manual(self, prompt: str, image: Optional[t.Tensor] = None, use_cache: Optional[bool] = False, max_new_tokens: int = 1, no_eos_token: Optional[bool] = False, **kwargs):
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

                new_token = self.tokenizer.decode(new_token_logits.argmax(-1))
                # if the new token is <end_of_sentence>, then we either 1) break, or 2) pick the second most likely token and continue generating
                if new_token == self.tokenizer.eos_token:
                    if not no_eos_token:
                        generated_tokens.append(new_token)
                        break
                    else:
                        second_largest_token = new_token_logits.topk(k=2, dim=-1, largest=True).indices[:, -1]
                        new_token = self.tokenizer.decode(second_largest_token)

                generated_tokens.append(new_token)

                past_output += new_token

        accumulated_final_logits = einops.rearrange(accumulated_final_logits, '(new_tokens batch) d_vocab -> new_tokens batch d_vocab', new_tokens=1)

        return accumulated_final_logits, generated_tokens

