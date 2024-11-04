import os
import json
import pandas as pd
from typing import Optional, Union, Tuple, Literal, Dict, List
from PIL import Image
from torchvision import transforms
import torch as t
import wandb
from config import DEVICE, PATH_TO_TENSORS
from torch.utils.data import DataLoader
from data import VLMJailbreakDataset, custom_collate_fn
from custom_image_transforms import CustomTransforms


class ControlSingleTokenAttack:
    def __init__(
        self, base_instance, cfg, wandb_name=None, wandb_logging=False
    ) -> None:
        self.base = base_instance
        self.cfg = cfg
        self.device = DEVICE
        self.wandb_logging = wandb_logging

        if self.wandb_logging:
            self.cfg.wandb_name = self.cfg.wandb_name or wandb_name
            wandb.init(
                project=self.cfg.wandb_project, name=self.cfg.wandb_name, config=self.cfg
            )

        # Create directory for saving tensors
        os.makedirs(PATH_TO_TENSORS, exist_ok=True)

    def train_attack(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        target: Optional[str] = None,
        verbose: Optional[bool] = True,
    ) -> Tuple[t.Tensor, t.Tensor, List[float]]:
        if target is None:
            target = self.cfg.single_token_target[0]

        init_image = transforms.ToTensor()(image).to(t.bfloat16).to(self.device)
        delta = t.zeros_like(
            init_image, dtype=t.bfloat16, requires_grad=True, device=self.device
        )

        target_tokenized = self.base.tokenizer.encode(
            target, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        # we will optimise the perturbation mask, not the original image
        optimizer = self.cfg.optimizer(
            [delta], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        eps = self.cfg.eps

        loss_train = []

        for step in range(self.cfg.n_epochs):
            optimizer.zero_grad()

            # should I normalise the perturbed image?
            perturbed_init_image = (
                (init_image + eps * delta.clamp(-1, 1)).clamp(0, 1).to(self.device)
                * 255.0
            )  # this will be a t.Tensor, not a PIL.Image.Image, but the range is [0.0, 255.0]

            output = self.base.generate_token_grad(prompt, perturbed_init_image)
            logits = output.logits  # [batch, sequence_position, vocab]
            next_token_logits = logits[:, -1]  # [batch, vocab]
            next_token_pred = self.base.tokenizer.batch_decode(
                next_token_logits.argmax(-1), skip_special_tokens=True
            )

            loss_fn = t.nn.CrossEntropyLoss()
            loss = loss_fn(next_token_logits, target_tokenized[:, -1])
            loss.backward()
            loss_train.append(loss.item())
            if self.wandb_logging:
                wandb.log({"train_loss": loss.item()}, step=step + 1)

            optimizer.step()

            if (step + 1) % (self.cfg.n_epochs // self.cfg.n_logs) == 0:
                next_token_probs = t.softmax(next_token_logits, dim=-1)
                target_token_prob = next_token_probs[0, target_tokenized.item()]
                if self.wandb_logging:
                    wandb.log(
                        {
                            "next_token_pred": next_token_pred,
                            "target_token_prob": target_token_prob.item(),
                        },
                        step=step + 1,
                    )
                if verbose:
                    print(f"Step {step+1}:")
                    print(f"loss: {loss.item():.4f}")
                    print(f"next_token_pred: {next_token_pred}")
                    print(f"target_token_prob: {target_token_prob.item():.4f}")
                    print("------------------")

            t.cuda.empty_cache()

        # Save tensors
        for tensor_name in ["init_image", "delta"]:
            t.save(
                eval(tensor_name), os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt")
            )
            if self.wandb_logging:
                wandb.save(
                    os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt"),
                    base_path="data_storage",
                )

        return init_image, delta, loss_train

    def execute_attack(
        self,
        prompt: Union[str, List[str]],
        image: Optional[t.Tensor] = None,
        delta: Optional[t.Tensor] = None,
        adversarial_image: Optional[t.Tensor] = None,
        eps: Optional[float] = None,
        generation_method: Optional[
            Literal["automatic_with_pil", "automatic", "manual"]
        ] = "automatic",
        max_new_tokens: Optional[int] = None,
        no_eos_token: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Dict, List[str]]:
        max_new_tokens = max_new_tokens or len(self.cfg.multi_token_target)
        eps = eps or self.cfg.eps

        if adversarial_image is None:
            if isinstance(delta, t.Tensor) and isinstance(image, t.Tensor):
                adversarial_image = image + eps * delta.clamp(-1, 1)
            else:
                raise ValueError(
                    "Both image and delta must be tensors when adversarial_image is not provided"
                )

        adversarial_image = adversarial_image * 255.0

        with t.no_grad():
            if generation_method == "automatic":
                output, answer = self.base.generate_autoregressive(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    **kwargs,
                )
            elif generation_method == "manual":
                output, answer = self.base.generate_autoregressive_manual(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Generation method {generation_method} not implemented yet."
                )

            if self.wandb_logging:
                wandb.log({"answer": answer})

        return output, answer

    def finish_wandb_logging(self):
        wandb.finish()


class ControlMultipleTokensAttack:
    def __init__(
        self,
        base_instance,
        cfg,
        wandb_run_id: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_logging: Optional[bool] = False,
    ) -> None:
        # if a wandb_entity is provided, a previous run will be used instead of starting a new one
        # this is useful e.g. when we want to load a previously obtained perturbation
        # and carry out/evaluate attacks using it
        self.base = base_instance
        self.cfg = cfg
        self.wandb_name = wandb_name or self.cfg.wandb_name
        self.wandb_run_id = wandb_run_id
        self.device = DEVICE
        self.wandb_logging = wandb_logging
        if self.wandb_logging:
            self._init_wandb()

    def _init_wandb(self) -> None:
        run = wandb.init(
            project=self.cfg.wandb_project,
            id=self.wandb_run_id,
            name=self.wandb_name,
            config=self.cfg,
            resume="allow",
        )
        print(f"Current run path: {run.path}")
        self.run = run

    # def _get_artifact(self, artifact_name) -> wandb.Artifact:
    #     try:
    #         artifact = self.run.use_artifact(f'{artifact_name}:latest')
    #         print(f"Using existing artifact: {artifact_name}")
    #     except wandb.errors.CommError:
    #         # If the artifact doesn't exist, create a new one
    #         artifact = wandb.Artifact(name=artifact_name, type='dataset')
    #         print(f"Created new artifact: {artifact_name}")
    #     return artifact

    def finish_wandb(self) -> None:
        wandb.finish()

    def _initialize_delta(
        self, image: Optional[Image.Image]
    ) -> Tuple[Optional[t.Tensor], t.Tensor]:
        if isinstance(image, Image.Image):
            init_image = transforms.ToTensor()(image).to(t.bfloat16).to(self.device)
            delta = t.zeros_like(
                init_image, dtype=t.bfloat16, requires_grad=True, device=self.device
            )
        elif image is None:
            init_image = None
            delta = t.rand(
                (3, self.cfg.image_size, self.cfg.image_size),
                dtype=t.bfloat16,
                requires_grad=True,
                device=self.device,
            )
        else:
            raise ValueError(f"Image type {type(image)} not implemented yet.")
        return init_image, delta

    def _get_perturbed_image(
        self,
        init_image: Optional[t.Tensor],
        delta: t.Tensor,
        eps: Optional[float] = None,
    ) -> t.Tensor:
        if init_image is not None:
            eps = eps or self.cfg.eps
            return (init_image + eps * delta.clamp(-1, 1)).clamp(0, 1).to(
                self.device
            ) * 255.0
        return (
            delta.clamp(0, 1) * 255.0
        )  # in this case, delta will be initialised as a tensor in the range [0, 1]

    def _forward_pass(
        self,
        prompt: Union[str, List[str]],
        perturbed_image: t.Tensor,
        target: List[str],
        training_method: str,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[t.Tensor, Dict, Dict]:
        loss_fn = t.nn.CrossEntropyLoss()
        loss = t.tensor(0.0).to(self.device)
        next_token_preds = {}
        target_token_probs = {}
        past_key_values = None
        past_output = ""
        next_token_pred = ""

        for ii, target_tok in enumerate(target):
            if ii > 0:
                token_to_append = self._get_token_to_append(
                    next_token_pred, target, ii, training_method
                )
                past_output += token_to_append

            output = self.base.generate_token_grad(
                prompt,
                perturbed_image,
                use_cache=use_cache,
                past_key_values=past_key_values,
                past_output=past_output,
                **kwargs,
            )
            past_key_values = output.past_key_values

            logits = output.logits
            next_token_logits = logits[:, -1]
            next_token_pred = self.base.tokenizer.decode(
                next_token_logits.argmax(-1), skip_special_tokens=True
            )

            target_tokenized = self.base.tokenizer.encode(
                target_tok, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            loss += loss_fn(next_token_logits, target_tokenized[:, -1])

            next_token_probs = t.softmax(next_token_logits, dim=-1)
            # next token prediction
            next_token_prob = next_token_probs[0].max()
            next_token_preds[next_token_pred] = f"{next_token_prob.item():.{5}e}"
            # target token probability
            target_token_prob = next_token_probs[0, target_tokenized.item()]
            target_token_probs[target_tok] = f"{target_token_prob.item():.{5}e}"

        return loss, next_token_preds, target_token_probs

    def _get_token_to_append(
        self, next_token_pred: str, target: List[str], index: int, training_method: str
    ) -> str:
        if training_method == "autoregressive":
            return next_token_pred
        elif training_method == "teacher_forcing":
            return target[index - 1]
        else:
            raise NotImplementedError(
                f"Method {training_method} not implemented. Use 'autoregressive' or 'teacher_forcing'"
            )

    def _save_tensors(self, init_image: Optional[t.Tensor], delta: t.Tensor) -> None:
        os.makedirs(PATH_TO_TENSORS, exist_ok=True)
        for tensor_name, tensor in [("init_image", init_image), ("delta", delta)]:
            if tensor is not None:
                t.save(tensor, os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt"))
                if self.wandb_logging:
                    wandb.save(
                        os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt"),
                        base_path="data_storage",
                )

    def train_attack(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, None],
        target: List[str],
        training_method: Literal[
            "autoregressive", "teacher_forcing"
        ] = "autoregressive",
        use_cache: Optional[bool] = False,
        eps: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[Optional[t.Tensor], t.Tensor, List[float]]:
        assert (
            self.cfg.n_epochs > self.cfg.n_logs
        ), "For MultiTokenAttack, n_epochs must be greater than n_logs."

        if self.wandb_logging:
            wandb.log({"training_method": training_method})

        init_image, delta = self._initialize_delta(image)
        optimizer = self.cfg.optimizer(
            [delta], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        loss_train: List[float] = []

        for step in range(self.cfg.n_epochs):
            optimizer.zero_grad()
            perturbed_image = self._get_perturbed_image(init_image, delta, eps)

            loss, next_token_preds, target_token_probs = self._forward_pass(
                prompt=prompt,
                perturbed_image=perturbed_image,
                target=target,
                training_method=training_method,
                use_cache=use_cache,
            )
            loss.backward()
            loss_train.append(loss.item())

            if self.wandb_logging:
                wandb.log({"train_loss": loss.item()}, step=step + 1)
            optimizer.step()

            if (step + 1) % (self.cfg.n_epochs // self.cfg.n_logs) == 0:
                if self.wandb_logging:
                    wandb.log(
                        {
                            "next_token_preds": next_token_preds,
                        "target_token_probs": target_token_probs,
                    },
                        step=step + 1,
                    )
                if verbose:
                    print(f"Step {step+1}:")
                    print(f"loss: {loss.item():.4f}")
                    print(f"next_token_preds: {next_token_preds}")
                    print(f"target_token_probs: {target_token_probs}")
                    print("------------------")

            t.cuda.empty_cache()

        self._save_tensors(init_image, delta)
        return init_image, delta, loss_train

    def execute_attack(
        self,
        prompt: Union[str, List[str]],
        image: Optional[t.Tensor] = None,
        delta: Optional[t.Tensor] = None,
        eps: Optional[float] = None,
        generation_method: str = "automatic",
        max_new_tokens: int = 10,
        no_eos_token: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Dict, List[str]]:
        eps = eps or self.cfg.eps

        if image is None and isinstance(delta, t.Tensor):
            adversarial_image = delta
        elif isinstance(delta, t.Tensor) and isinstance(image, t.Tensor):
            adversarial_image = image + eps * delta.clamp(-1, 1)
        else:
            raise NotImplementedError("""You can either supply the adversarial image through 'delta', or the original image and its perturbation 'delta'.
            Both need to be a t.Tensor.""")

        adversarial_image = adversarial_image.clamp(0, 1)

        if generation_method == "automatic_with_pil":
            adversarial_image = transforms.ToPILImage()(
                adversarial_image.detach().cpu().float()
            )
        else:
            adversarial_image = adversarial_image * 255.0

        with t.no_grad():
            if generation_method == "automatic_with_pil":
                output, answer = self.base.generate_autoregressive_with_pil(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                    **kwargs,
                )
            elif generation_method == "automatic":
                output, answer = self.base.generate_autoregressive(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                    **kwargs,
                )
            elif generation_method == "manual":
                output, answer = self.base.generate_autoregressive_manual(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                )
            else:
                raise NotImplementedError(
                    f"Generation method {generation_method} not implemented yet."
                )

            if self.wandb_logging:
                wandb.log({"generation_method": generation_method})
                wandb.log({"answer": answer})

        return output, answer


class JailbreakAttack(ControlMultipleTokensAttack):
    def __init__(
        self,
        base_instance,
        cfg,
        wandb_run_id: Optional[str] = None,
        wandb_name: Optional[str] = None,
    ) -> None:
        super().__init__(base_instance, cfg, wandb_run_id, wandb_name)

    # overwrites the method from ControlMultipleTokensAttack
    def _initialize_delta(
        self, images: Optional[List[Image.Image]]
    ) -> Tuple[Optional[List[t.Tensor]], t.Tensor]:
        if isinstance(images, list):
            init_images = [
                transforms.ToTensor()(img).to(t.bfloat16).to(self.device)
                for img in images
            ]
            if len(set([x.shape for x in init_images])) == 1:
                delta = t.zeros_like(
                    init_images[0],
                    dtype=t.bfloat16,
                    requires_grad=True,
                    device=self.device,
                )
            else:
                raise ValueError("All images need to have the same shape!")
        elif images is None:
            init_images = None
            delta = t.rand(
                (3, self.cfg.image_size, self.cfg.image_size),
                dtype=t.bfloat16,
                requires_grad=True,
                device=self.device,
            )
        else:
            raise ValueError(f"Image type {type(images)} not implemented yet.")
        return init_images, delta

    # overwrites the method from ControlMultipleTokensAttack
    def _get_perturbed_images(
        self,
        init_images: Optional[List[t.Tensor]],
        delta: t.Tensor,
        eps: Optional[float] = None,
    ) -> Union[List[t.Tensor], t.Tensor]:
        if init_images is not None:
            eps = eps or self.cfg.eps
            perturbed_images = [
                (img + eps * delta.clamp(-1, 1)).clamp(0, 1).to(self.device) * 255.0
                for img in init_images
            ]
            return perturbed_images
        return (
            delta.clamp(0, 1) * 255.0
        )

    # overwrites the method from ControlMultipleTokensAttack
    def _save_tensors(self, delta: t.Tensor) -> None:
        os.makedirs(PATH_TO_TENSORS, exist_ok=True)
        for tensor_name, tensor in [("delta", delta)]:
            t.save(tensor, os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt"))
            if self.wandb_logging:
                wandb.save(
                    os.path.join(PATH_TO_TENSORS, f"{tensor_name}.pt"),
                    base_path="data_storage",
                )

    def _assertions(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]],
        targets: Optional[Union[List[List[str]], List[str]]],
        batch_size: int,
    ) -> None:
        assert batch_size <= len(
            prompts
        ), "Batch size cannot be larger than the number of prompts."

        if images is not None:
            assert (
                len(images) == len(prompts)
            ), "If supplying image to be perturbed, number of prompts and images must be the same."
            n_images = len(images)
            print(f"Number of images: {n_images}")

        if targets is None:
            raise ValueError(
                "Target cannot be None. Set it to either a list of strings (tokens for a single target) or a list of list of strings (tokens for a set of targets)"
            )
        elif isinstance(targets[0], list):
            assert (
                len(targets) == len(prompts)
            ), "For the multi-target case, number of prompts and targets must be the same."
            n_targets = len(targets)
            print(f"Number of targets: {n_targets}")

    def train(
        self,
        prompts: List[str],
        images: Union[List[Image.Image], None],
        targets: List[List[str]],
        training_method: Literal[
            "autoregressive", "teacher_forcing"
        ] = "autoregressive",
        use_cache: Optional[bool] = False,
        eps: Optional[float] = None,
        verbose: bool = True,
        batch_size: int = 8,
        augmentations: Optional[dict] = None,
    ) -> Tuple[t.Tensor, List[float]]:
        eps = eps or self.cfg.eps
        self._assertions(prompts, images, targets, batch_size)
        if self.wandb_logging:
            wandb.log({"training_method": training_method})

        init_images, delta = self._initialize_delta(images)

        dataset = VLMJailbreakDataset(
            prompts=prompts, images=init_images, targets=targets
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
        )

        n_batches = len(dataloader)
        print(f"Number of batches: {n_batches}")

        if augmentations:
            print('Using augmentations')
            transforms = CustomTransforms(**augmentations)

        assert (
            (self.cfg.n_epochs * n_batches) > self.cfg.n_logs
        ), "For JailbreakAttack, n_epochs*n_batches must be greater than n_logs."

        optimizer = self.cfg.optimizer(
            [delta], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        loss_train: List[float] = []

        current_iter = 1
        for step in range(self.cfg.n_epochs):
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()  # <-- needs to be inside the 'batch loop'

                batch_prompts, batch_init_images, batch_targets = batch
                perturbed_images = self._get_perturbed_images(
                    batch_init_images, delta, eps
                )  # Union[List[t.Tensor], t.Tensor]
                # output is in the range [0.0, 255.0], not [0.0, 1.0]

                loss_batch = t.tensor(0.0).to(self.device)

                if not isinstance(perturbed_images, (list, t.Tensor)):
                    raise NotImplementedError(
                        f"Perturbed images type {type(perturbed_images)} not implemented yet."
                    )

                if not isinstance(batch_targets[0], (list, str)):
                    raise ValueError(
                        "Targets must be either a list of strings (tokens for a single target) or a list of list of strings (tokens for a set of targets)"
                    )

                if augmentations:
                    if isinstance(perturbed_images, list):
                        perturbed_images = [transforms(img) for img in perturbed_images]
                    else:
                        perturbed_images = transforms(perturbed_images)

                is_multi_image = isinstance(perturbed_images, list)
                is_multi_target = isinstance(batch_targets[0], list)

                next_token_preds_batch = {}
                target_token_probs_batch = {}

                for ii, prompt in enumerate(batch_prompts):
                    perturbed_image: t.Tensor = (
                        perturbed_images[ii] if is_multi_image else perturbed_images
                    )
                    target = batch_targets[ii] if is_multi_target else batch_targets
                    loss, next_token_preds, target_token_probs = self._forward_pass(
                        prompt,
                        perturbed_image,
                        target,
                        training_method,
                        use_cache=use_cache,
                    )
                    loss_batch += loss
                    next_token_preds_batch[ii] = next_token_preds
                    target_token_probs_batch[ii] = target_token_probs

                loss_batch.backward()
                loss_train.append(loss_batch.item())

                if self.wandb_logging:
                    wandb.log({"train_loss": loss_batch.item()}, step=current_iter)
                    wandb.log(
                        {
                            "next_token_preds_batch": json.dumps(next_token_preds_batch),
                            "target_token_probs_batch": json.dumps(target_token_probs),
                        },
                        step=current_iter,
                    )

                optimizer.step()

                if (
                    current_iter % ((self.cfg.n_epochs * n_batches) // self.cfg.n_logs)
                    == 0
                ):
                    if verbose:
                        print(
                            f"Current iter {current_iter}, Step {step+1}, Batch {batch_idx+1}:"
                        )
                        print(f"next_token_preds_batch: {next_token_preds_batch}")
                        print(f"target_token_probs_batch: {target_token_probs_batch}")
                        print(f"loss: {loss_batch.item():.4f}")
                        print("------------------")

                t.cuda.empty_cache()

                current_iter += 1

        self._save_tensors(delta)
        return delta, loss_train

    def execute_with_clamp(
        self,
        prompt: Union[str, List[str]],
        image: Optional[t.Tensor] = None,
        delta: Optional[t.Tensor] = None,
        eps: Optional[float] = None,
        generation_method: str = "automatic",
        max_new_tokens: Optional[int] = None,
        no_eos_token: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Dict, List[str]]:
        return super().execute_attack(
            prompt,
            image,
            delta,
            eps,
            generation_method,
            max_new_tokens,
            no_eos_token,
            use_cache,
            **kwargs,
        )

    # overwrites the method from ControlMultipleTokensAttack
    def execute(
        self,
        prompt: Union[str, List[str]],
        image: Optional[t.Tensor] = None,
        delta: Optional[t.Tensor] = None,
        eps: Optional[float] = None,
        generation_method: Optional[
            Literal["automatic_with_pil", "automatic", "manual"]
        ] = "automatic",
        max_new_tokens: Optional[int] = None,
        no_eos_token: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Dict, List[str]]:
        max_new_tokens = max_new_tokens or len(self.cfg.multi_token_target)
        eps = eps or self.cfg.eps

        # the only difference between here and this function in the parents class is that
        # here we do not do adversarial_image.clamp(0, 1)
        if image is None and isinstance(delta, t.Tensor):
            adversarial_image = delta
        elif isinstance(delta, t.Tensor) and isinstance(image, t.Tensor):
            adversarial_image = image + eps * delta.clamp(-1, 1)
        else:
            raise NotImplementedError("""You can either supply the adversarial image through 'delta', or the original image and its perturbation 'delta'.
            Both need to be a t.Tensor.""")

        if generation_method == "automatic_with_pil":
            adversarial_image = transforms.ToPILImage()(
                adversarial_image.detach().cpu().float()
            )  # <-- check clamping here
        else:
            adversarial_image = adversarial_image * 255.0  # <-- and here

        with t.no_grad():
            if generation_method == "automatic_with_pil":
                output, answer = self.base.generate_autoregressive_with_pil(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                    **kwargs,
                )
            elif generation_method == "automatic":
                output, answer = self.base.generate_autoregressive(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                    **kwargs,
                )
            elif generation_method == "manual":
                output, answer = self.base.generate_autoregressive_manual(
                    prompt,
                    adversarial_image,
                    max_new_tokens=max_new_tokens,
                    no_eos_token=no_eos_token,
                    use_cache=use_cache,
                )
            else:
                raise NotImplementedError(
                    f"Generation method {generation_method} not implemented yet."
                )

            if self.wandb_logging:
                wandb.log({"generation_method": generation_method})
                wandb.log({"answer": answer})

        return output, answer

    def test(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        delta: Optional[t.Tensor] = None,
        eps: Optional[float] = None,
        generation_method: Union[
            List[str], Literal["automatic_with_pil", "automatic", "manual", "all"]
        ] = "all",
        **kwargs,
    ) -> Dict:
        if images is not None:
            assert (
                len(images) == len(prompts)
            ), "If supplying images to be perturbed, number of prompts and images must be the same."
            n_images = len(images)
            print(f"Number of images: {n_images}")
            init_images = [
                transforms.ToTensor()(img).to(t.bfloat16).to(self.device)
                for img in images
            ]
        else:
            init_images = [None for _ in range(len(prompts))]

        if generation_method == "all":
            generation_method = ["automatic_with_pil", "automatic", "manual"]
        elif isinstance(generation_method, str):
            generation_method = [generation_method]

        answers_dict = {}
        for gen_method in generation_method:
            answers = []
            for prompt, init_image in zip(prompts, init_images):
                # _, answer = self.execute(prompt, init_image, delta, eps, gen_method, **kwargs)
                _, answer = self.execute_with_clamp(
                    prompt, init_image, delta, eps, gen_method, **kwargs
                )  # <-- clamping
                if isinstance(answer, list):
                    answer = "".join(answer)
                answers.append(answer)
            answers_dict[gen_method] = answers

            if self.wandb_logging:
                wandb.log({"answers_dict": answers_dict})
        return answers_dict

    def test_dataset(
        self, df_test: pd.DataFrame, delta: Optional[t.Tensor] = None, **kwargs
    ):
        prompts = df_test["goal"].to_list()
        images = df_test["image"].to_list() if "image" in df_test.columns else None

        answers_dict = self.test(prompts, images, delta, **kwargs)

        for gen_method in answers_dict.keys():
            df_test[gen_method] = answers_dict[gen_method]

        if self.wandb_logging:
            artifact = wandb.Artifact("jailbreak_data", type="jailbreak_data")
            table = wandb.Table(dataframe=df_test)
            artifact.add(table, name="jailbreak_completions")
            self.run.log_artifact(artifact)

        return table

    def eval_dataset(self, **kwargs):
        artifact = self.run.use_artifact("jailbreak_data:latest")
        artifact.download(path_prefix="jailbreak_completions")
        table = artifact.get("jailbreak_completions")
        df = pd.DataFrame(data=table.data, columns=table.columns)

        for method in df.columns.difference(["goal", "target"]):
            col_name = method + "_scores"
            for ind in df.index:
                target = df.loc[ind, "target"]
                jailbreak_output = df.loc[ind, method]

                prompt = self.cfg.jailbreak_eval_user_prompt.format(
                    target=target, text=jailbreak_output
                )
                _, answer = self.base.generate_autoregressive(
                    prompt=prompt,
                    image=None,
                    system_prompt=self.cfg.jailbreak_eval_system_prompt,
                    **kwargs,
                )

                df.loc[ind, col_name] = answer

            if self.wandb_logging:
                scores_artifact = wandb.Artifact("jailbreak_scores", type="jailbreak_scores")
                table = wandb.Table(dataframe=df)
                scores_artifact.add(table, name="jailbreak_scores")
                self.run.log_artifact(scores_artifact)

        return table
