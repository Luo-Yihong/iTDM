from collections import defaultdict
from copy import deepcopy
from concurrent import futures
from functools import partial
import datetime
import hashlib
import json
import os
import random
import tempfile
import time

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration, DistributedDataParallelKwargs
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import tqdm
import wandb

import flow_grpo.rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/itdm.py:geneval_sd3_8gpu", "Training configuration.")

logger = get_logger(__name__)


def generate_shared_sampled_timesteps(accelerator, train_timesteps, M):
    """Sample M timestep indices on rank 0 and broadcast them so all ranks train on the same ones."""
    device = accelerator.device
    if accelerator.is_main_process:
        sampled = random.sample(train_timesteps, M)
        sampled_tensor = torch.tensor(sampled, device=device, dtype=torch.long)
    else:
        sampled_tensor = torch.empty(M, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(sampled_tensor, src=0)
    return sampled_tensor.tolist()


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    """Yield per-rank batches such that every prompt appears exactly k times across
    `batches_per_epoch` consecutive global batches (i.e. complete groups of size k)."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, batches_per_epoch=1):
        self.dataset = dataset
        self.batch_size = batch_size  # per replica
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.total_samples = self.num_replicas * self.batch_size * batches_per_epoch
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # number of unique prompts
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch // self.batches_per_epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = (self.epoch % self.batches_per_epoch * self.num_replicas + i) * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(repeated_indices[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def predict_v(transformer, noisy_samples, timesteps, embeds, pooled_embeds,
              cfg_scale=1.0, neg_embeds=None, neg_pooled_embeds=None):
    """Predict velocity, with CFG if `cfg_scale > 1`. The unconditional branch is detached."""
    if cfg_scale <= 1:
        return transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    noise_pred = transformer(
        hidden_states=torch.cat([noisy_samples] * 2),
        timestep=torch.cat([timesteps] * 2),
        encoder_hidden_states=torch.cat([neg_embeds, embeds]),
        pooled_projections=torch.cat([neg_pooled_embeds, pooled_embeds]),
        return_dict=False,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred_uncond = noise_pred_uncond.detach()
    return noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)


def initialize_itdm_adapters(base_transformer, lora_config, lora_path="", dsm_with_default=False):
    """Load the policy (`default`) and DSM (`dsm`) adapters, or initialize DSM from the policy."""
    if lora_path:
        model = PeftModel.from_pretrained(base_transformer, lora_path, is_trainable=True)
    else:
        model = get_peft_model(base_transformer, lora_config)
    dsm_path = os.path.join(lora_path, "dsm") if lora_path else ""
    if dsm_path and os.path.isfile(os.path.join(dsm_path, "adapter_config.json")):
        model.load_adapter(dsm_path, adapter_name="dsm", is_trainable=True)
    else:
        dsm_config = deepcopy(model.peft_config["default"])
        dsm_config.inference_mode = False
        model.add_adapter("dsm", dsm_config)
        policy_state = get_peft_model_state_dict(model, adapter_name="default")
        set_peft_model_state_dict(model, policy_state, adapter_name="dsm")
    return ITDMModel(model, dsm_with_default=dsm_with_default)


class ITDMModel(torch.nn.Module):
    """Route forwards to the policy or DSM adapter.

    With `dsm_with_default=True`, the DSM forward stacks the (frozen) policy adapter and the
    trainable DSM adapter; otherwise DSM is a standalone adapter on top of the base model.
    """

    def __init__(self, transformer, dsm_with_default=False):
        super().__init__()
        self.transformer = transformer
        self.dsm_with_default = dsm_with_default
        transformer.set_adapter("default")
        self.policy_parameters = tuple(p for p in transformer.parameters() if p.requires_grad)
        transformer.set_adapter("dsm")
        self.dsm_parameters = tuple(p for p in transformer.parameters() if p.requires_grad)
        if not self.policy_parameters or not self.dsm_parameters:
            raise ValueError("ITDM requires nonempty default and dsm adapters")
        if {id(p) for p in self.policy_parameters} & {id(p) for p in self.dsm_parameters}:
            raise ValueError("Policy and DSM adapters must not share trainable parameters")
        self.set_adapter("default")

    def set_adapter(self, name):
        self.transformer.set_adapter(name)
        combined_dsm = name == "dsm" and self.dsm_with_default
        if combined_dsm:
            # PeftModel.set_adapter accepts one name; its LoRA tuner supports a list.
            self.transformer.base_model.set_adapter(["default", "dsm"])
        # Both parameter sets stay registered with DDP; find_unused_parameters handles
        # parameters that are inactive or frozen in a given forward.
        for p in self.policy_parameters + self.dsm_parameters:
            p.requires_grad_(True)
        if combined_dsm:
            for p in self.policy_parameters:
                p.requires_grad_(False)

    def forward(self, xt, t, embeds, pooled_embeds, adapter, cfg_scale=1.0, neg_embeds=None, neg_pooled_embeds=None):
        """The DSM adapter is always CFG-free; `cfg_scale` only applies to the policy."""
        self.set_adapter(adapter)
        if adapter == "dsm":
            cfg_scale = 1.0
        return predict_v(
            self.transformer, xt, t, embeds, pooled_embeds,
            cfg_scale=cfg_scale, neg_embeds=neg_embeds, neg_pooled_embeds=neg_pooled_embeds,
        )


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Return the fraction of unique prompts whose reward std is zero, and the mean per-prompt std."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_ratio = np.count_nonzero(prompt_std_devs == 0) / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        generators.append(torch.Generator().manual_seed(seed))
    return generators


def evaluate(
    pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step,
    reward_fn, executor, autocast, ema, transformer_trainable_parameters,
):
    """Evaluate the (EMA) policy with rollout CFG and log rewards and sample images."""
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    try:
        neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
            [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
        sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

        all_rewards = defaultdict(list)
        for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            prompts, prompt_metadata = test_batch
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            # The last batch may not be a full batch.
            if len(prompt_embeds) < len(sample_neg_prompt_embeds):
                sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
                sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
            with autocast():
                with torch.no_grad():
                    images, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=0,
                    )
            images = images.float()
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)  # yield so that reward computation starts
            rewards, reward_metadata = rewards.result()
            for key, value in rewards.items():
                rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards[key].append(rewards_gather)

        last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
        last_batch_prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
        last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
            last_batch_prompt_ids_gather, skip_special_tokens=True
        )
        last_batch_rewards_gather = {
            key: accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            for key, value in rewards.items()
        }

        all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
        if accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(last_batch_images_gather))
                for idx in range(num_samples):
                    image = last_batch_images_gather[idx]
                    pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = last_batch_prompts_gather[:num_samples]
                sampled_rewards = [
                    {k: last_batch_rewards_gather[k][idx] for k in last_batch_rewards_gather}
                    for idx in range(num_samples)
                ]
                wandb.log(
                    {
                        "eval_images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.1000} | "
                                + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                            )
                            for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                        **{
                            f"eval_reward_{key}": float(np.mean(value[value != -10]))
                            for key, value in all_rewards.items()
                        },
                    },
                    step=global_step,
                )
    finally:
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model.transformer if isinstance(model, ITDMModel) else model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    """Save `ema.ckpt` (if EMA is enabled) and the policy LoRA (`lora/`) plus DSM LoRA (`lora/dsm/`). Main process only."""
    if not accelerator.is_main_process:
        return
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if config.train.ema:
        ema.save(os.path.join(save_root, "ema.ckpt"))
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    try:
        unwrap_model(transformer, accelerator).save_pretrained(
            save_root_lora, selected_adapters=["default", "dsm"],
        )
    finally:
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


def main(_):
    config = FLAGS.config
    if not config.use_lora:
        raise ValueError("ITDM requires use_lora=True (policy and DSM adapters)")
    ttur = config.train.ttur
    if type(ttur) is not int or ttur < 1:
        raise ValueError("config.train.ttur must be a positive integer")
    max_policy_steps = config.train.max_policy_steps

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = config.num_train_timesteps

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # We accumulate over samples and over the sampled timesteps of each sample.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.is_main_process:
        wandb.init(
            project="itdm",
            name=config.run_name,
            config=config.to_dict(),
            dir=config.logdir,
        )
    logger.info(f"\n{config}")

    # device_specific seeding gives different prompts on different devices
    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="scheduler"
    )
    pipeline.scheduler.config['flow_shift'] = config.flow_shift
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Non-trainable weights are only used for inference, so cast them to half precision.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    transformer_lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ],
    )
    training_model = initialize_itdm_adapters(
        pipeline.transformer, transformer_lora_config, config.train.lora_path,
        dsm_with_default=config.train.dsm_with_default,
    )
    pipeline.transformer = training_model.transformer

    transformer = training_model
    transformer_trainable_parameters = list(training_model.policy_parameters)
    dsm_trainable_parameters = list(training_model.dsm_parameters)
    # EMA of the policy used for evaluation and checkpoints.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    # Fast EMA of the policy used for rollouts after `config.switch_ema_ref` steps (never if negative).
    ema_ref = EMAModuleWrapper(transformer_trainable_parameters, decay=0.3, update_step_interval=1, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    dsm_optimizer = optimizer_cls(
        dsm_trainable_parameters,
        lr=config.train.dsm_learning_rate,
        betas=(config.train.dsm_adam_beta1, config.train.dsm_adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        dataset_cls, test_num_workers = TextPromptDataset, 8
    elif config.prompt_fn == "geneval":
        dataset_cls, test_num_workers = GenevalPromptDataset, 0
    else:
        raise NotImplementedError("Only general_ocr and geneval prompt_fn are supported")
    train_dataset = dataset_cls(config.dataset, 'train')
    test_dataset = dataset_cls(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        batches_per_epoch=config.sample.group_batches,
    )
    # The sampler's epoch is set before every batch, so the DataLoader must not prefetch (num_workers=0).
    # It is also not passed to `accelerator.prepare`, which would shard it again.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=dataset_cls.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=dataset_cls.collate_fn,
        shuffle=False,
        num_workers=test_num_workers,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = accelerator.autocast

    transformer, optimizer, dsm_optimizer, test_dataloader = accelerator.prepare(
        transformer, optimizer, dsm_optimizer, test_dataloader,
    )

    # Rewards are computed asynchronously while sampling continues.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0  # Completed DSM updates; drives the TTUR schedule.
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        training_model.set_adapter("default")
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            evaluate(
                pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step,
                reward_fn, executor, autocast, ema, transformer_trainable_parameters,
            )
        if epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        use_ema_rollout = config.switch_ema_ref >= 0 and global_step > config.switch_ema_ref
        if use_ema_rollout:
            ema_ref.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents_list = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                    )

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            images = images.float()
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)  # yield so that reward computation starts

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "x0": latents_list[-1],
                    "rewards": rewards,
                }
            )
        if use_ema_rollout:
            ema_ref.copy_temp_to(transformer_trainable_parameters)

        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into a dict where each entry has shape (num_batches_per_epoch * batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # save as JPEG to keep wandb uploads small
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)
                for idx, i in enumerate(sample_indices):
                    pil = Image.fromarray(
                        (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]
                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Repeat along the timestep dimension so that timestep-dependent advantages are easy to add.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.num_steps)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if '_strict_accuracy' not in key and '_accuracy' not in key
                    },
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; keep only the entries of this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)

            train_timesteps = list(range(config.train.max_timestep_index + 1))
            sampled_timesteps = generate_shared_sampled_timesteps(accelerator, train_timesteps, num_train_timesteps)
            microsteps = [(sample, j) for sample in samples_batched for j in sampled_timesteps]
            accumulation_steps = accelerator.gradient_accumulation_steps
            if len(microsteps) % accumulation_steps:
                raise ValueError("ITDM needs complete accumulation windows for DSM then policy updates")

            for start in tqdm(
                range(0, len(microsteps), accumulation_steps),
                desc=f"Epoch {epoch}.{inner_epoch}: DSM / policy every {ttur} steps",
                disable=not accelerator.is_local_main_process,
            ):
                window = microsteps[start:start + accumulation_steps]
                noises = [torch.randn_like(sample["x0"]) for sample, _ in window]

                # One DSM update per accumulation window: fit the DSM adapter to the current rollouts.
                for (sample, j), noise_diffuse in zip(window, noises):
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]
                    with accelerator.accumulate(transformer):
                        x0 = sample["x0"].detach()
                        t = sample["timesteps"][:, j]
                        sigmas_t = (t / 1000).reshape(-1, 1, 1, 1)
                        xt = (1 - sigmas_t) * x0 + sigmas_t * noise_diffuse
                        target_v = (noise_diffuse - x0).float()

                        with autocast():
                            online_v = transformer(xt, t, embeds, pooled_embeds, "dsm")
                        loss = (target_v - online_v.float()).square().mean()
                        info["online_dsm_loss"].append(loss.detach())

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.unscale_gradients(dsm_optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                dsm_trainable_parameters, config.train.max_grad_norm,
                            )
                            info["dsm_grad_norm"].append(grad_norm.detach())
                        dsm_optimizer.step()
                        dsm_optimizer.zero_grad(set_to_none=True)

                # One policy update every `ttur` DSM updates, on the same window and noises.
                if (global_step + 1) % ttur == 0:
                    for (sample, j), noise_diffuse in zip(window, noises):
                        embeds = sample["prompt_embeds"]
                        pooled_embeds = sample["pooled_prompt_embeds"]
                        cfg_kwargs = dict(
                            cfg_scale=config.sample.guidance_scale,
                            neg_embeds=train_neg_prompt_embeds[:len(embeds)],
                            neg_pooled_embeds=train_neg_pooled_prompt_embeds[:len(pooled_embeds)],
                        )
                        with accelerator.accumulate(transformer):
                            x0 = sample["x0"].detach()
                            t = sample["timesteps"][:, j]
                            sigmas_t = (t / 1000).reshape(-1, 1, 1, 1)
                            xt = (1 - sigmas_t) * x0 + sigmas_t * noise_diffuse
                            target_v = (noise_diffuse - x0).float()

                            # Close this autocast context before the trainable forward so that
                            # cached casts of frozen weights are not reused for policy gradients.
                            with torch.no_grad(), autocast():
                                online_v = training_model(xt, t, embeds, pooled_embeds, "dsm").float()
                                training_model.set_adapter("default")
                                with pipeline.transformer.disable_adapter():
                                    ref_v = predict_v(
                                        pipeline.transformer, xt, t, embeds, pooled_embeds, **cfg_kwargs,
                                    ).float()
                            with autocast():
                                model_v = transformer(xt, t, embeds, pooled_embeds, "default", **cfg_kwargs).float()
                            # Clipping slightly violates sum_G A = 0 but rarely triggers in practice.
                            advantages = torch.clamp(
                                sample["advantages"][:, j],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )
                            r_normalized = advantages / 5
                            # Reward-Driven Revision of the velocity
                            model_v_revised = (model_v - r_normalized.view(-1, 1, 1, 1) * (online_v - target_v)).detach()
                            itdm_policy_loss = (model_v - model_v_revised).square().mean()
                            if config.train.beta > 0:
                                kl_loss = (model_v - ref_v).square().mean()
                                info["kl_loss"].append(kl_loss.detach())
                                loss = itdm_policy_loss + config.train.beta * kl_loss
                            else:
                                loss = itdm_policy_loss

                            info["itdm_policy_loss"].append(itdm_policy_loss.detach())
                            info["online_dsm_loss"].append((target_v - online_v).square().mean())

                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.unscale_gradients(optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    transformer_trainable_parameters, config.train.max_grad_norm,
                                )
                                info["policy_grad_norm"].append(grad_norm.detach())
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                    policy_step = (global_step + 1) // ttur
                    ema_ref_decay = min(0.3, 0.001 * (policy_step - 1))
                    ema_ref.step(transformer_trainable_parameters, policy_step - 1, decay=ema_ref_decay)
                    if config.train.ema:
                        ema.step(transformer_trainable_parameters, policy_step)

                training_model.set_adapter("default")
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                info.update({
                    "epoch": epoch, "inner_epoch": inner_epoch,
                    "dsm_step": global_step + 1, "policy_step": (global_step + 1) // ttur,
                })
                if accelerator.is_main_process:
                    wandb.log(info, step=global_step)
                global_step += 1

                if max_policy_steps and global_step // ttur >= max_policy_steps:
                    accelerator.wait_for_everyone()
                    save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
                    accelerator.wait_for_everyone()
                    evaluate(
                        pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step,
                        reward_fn, executor, autocast, ema, transformer_trainable_parameters,
                    )
                    if accelerator.is_main_process:
                        wandb.finish()
                    return
                info = defaultdict(list)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
