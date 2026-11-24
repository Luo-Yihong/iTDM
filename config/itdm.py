import importlib.util
import os

_spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def itdm_sd3_8gpu():
    """Shared ITDM recipe for SD3.5-Medium on 8 GPUs: 24 groups x 24 images per rollout."""
    gpu_number = 8
    num_groups = 24  # prompts per rollout
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.resolution = 512
    config.save_freq = 60
    config.eval_freq = 60

    # Rollout: deterministic DPM-Solver with CFG.
    config.flow_shift = 3
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.0
    config.sample.same_latent = False
    config.sample.num_image_per_prompt = 24
    config.sample.train_batch_size = 8
    images_per_rollout = num_groups * config.sample.num_image_per_prompt
    global_batch_size = gpu_number * config.sample.train_batch_size
    assert images_per_rollout % global_batch_size == 0, "num_groups * group size must be divisible by the global batch size"
    config.sample.num_batches_per_epoch = images_per_rollout // global_batch_size
    config.sample.group_batches = config.sample.num_batches_per_epoch

    # Training: each epoch runs first M DSM updates, then M policy updates (never interleaved),
    # where M = num_batches_per_epoch / gradient_accumulation_steps (must divide evenly).
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch  # M = 1
    config.train.num_inner_epochs = 1
    config.num_train_timesteps = 4  # timesteps sampled per trajectory
    config.train.max_timestep_index = 8  # sample timestep indices from [0, max_timestep_index]
    # Run the policy phase once every `ttur` epochs (the DSM phase runs every epoch).
    config.train.ttur = 1
    # True: the DSM forward stacks the frozen policy adapter and the trainable DSM adapter.
    # False: DSM is a standalone adapter on the base model. DSM is always CFG-free.
    config.train.dsm_with_default = True
    # Weight of the velocity-MSE penalty to the frozen base model.
    config.train.beta = 1e-3
    config.train.adv_clip_max = 5
    config.train.ema = True
    config.train.lora_path = ""  # resume from `<ckpt>/lora` (DSM is loaded from `<ckpt>/lora/dsm` if present)
    # Stop, save, and evaluate after this many policy updates (0 = run forever).
    config.train.max_policy_steps = 3000
    config.switch_ema_ref = -1  # always roll out with the current policy
    # The DSM optimizer uses no momentum.
    config.train.dsm_adam_beta1 = 0.0
    return config


def geneval_sd3_8gpu():
    config = itdm_sd3_8gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.prompt_fn = "geneval"
    config.reward_fn = {"geneval": 1.0}
    config.sample.global_std = False
    # The test set has 2212 prompts; 8 GPUs x 14 keeps the padded last batch small.
    config.sample.test_batch_size = 14
    config.run_name = "itdm_geneval"
    config.logdir = config.save_dir = "logs/itdm/geneval"
    return config


def ocr_sd3_8gpu():
    config = itdm_sd3_8gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"ocr": 1.0}
    config.sample.global_std = True
    # The test set has 1018 prompts; 8 GPUs x 16 keeps the padded last batch small.
    config.train.learning_rate = 1e-4
    config.sample.test_batch_size = 16
    config.run_name = "itdm_ocr"
    config.logdir = config.save_dir = "logs/itdm/ocr"
    return config


def pickscore_sd3_8gpu():
    config = itdm_sd3_8gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"pickscore": 1.0}
    config.sample.global_std = True
    # The test set has 2048 prompts; 8 GPUs x 16 divides it evenly.
    config.sample.test_batch_size = 16
    config.run_name = "itdm_pickscore"
    config.logdir = config.save_dir = "logs/itdm/pickscore"
    return config


def get_config(name):
    return globals()[name]()
