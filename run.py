import os
import copy
import json
import argparse

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.train import train
from utils.optim.factory import get_optimizer, get_scheduler
from utils.optim.muon2 import Muon2
from utils.utils import hash_config, set_seed, get_worker_info
from utils.model import load_model
from utils.dataloader import ShardedDataLoader, get_dataset_path

parser = argparse.ArgumentParser(description='Pre-train a GPT model with Muon^2 or a baseline optimizer.')
parser.add_argument('--config', type=str, required=True, help='Path to a YAML config file')
parser.add_argument('--prefix', type=str, default='', help='Prefix for the W&B run name')
parser.add_argument('--suffix', type=str, default='', help='Suffix for the W&B run name and the results file')
args = parser.parse_args()
set_seed(42)

# First set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
print(f"Using device: {device}")

# Parse config
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
# e.g. configs/gpt_small/muon2_ns5.yaml -> gpt_small/muon2_ns5
outputname = config_file.split("configs/")[-1] if "configs/" in config_file else os.path.basename(config_file)
outputname = outputname.removesuffix(".yaml")
output_dir = f"{config['logging_params']['results_dir']}/{outputname}"
CKPT_DIR = config['logging_params']['ckpt_dir']
ckpt_dir_base = CKPT_DIR + f"/{outputname}/" if CKPT_DIR != "" else ""
if master_process:
    print(f"Loading configuration from {config_file}")
    print(f"Training on dataset {config['dataset']['name']}")
    os.makedirs(output_dir, exist_ok=True)
    if CKPT_DIR != "": os.makedirs(ckpt_dir_base, exist_ok=True)

# Load model
model = load_model(config['model_config'])
if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(model)

training_params = config['training_params']
list_optimizer_params = config["optimizer_params"]
torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

# Load data
tokenizer_name = config['dataset'].get('tokenizer', 'gpt2')
dataset_path = get_dataset_path(config['dataset']['name'], tokenizer_name)
if master_process: print(f"Load data from {dataset_path}")
B, T = training_params['batch_size'], training_params['context_length']
assert training_params['tokens_processed'] % (world_size * B * T) == 0
# Limit the number of training shards (each shard is 0.1B tokens)
num_shards = None
if 'total_tokens_B' in config['dataset']:
    num_shards = int(config['dataset']['total_tokens_B'] * 10)
    if master_process: print(f"Limiting training to {config['dataset']['total_tokens_B']}B tokens ({num_shards} shards)")
train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device, num_shards=num_shards)
val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
if master_process:
    print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
    print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
    print(f"Total number of iterations : {total_iterations}")

# Loop over optimizers and learning rates
for opt_config in list_optimizer_params:
    for lr in opt_config['lr']:
        if master_process:
            print(f"\nTraining with optimizer {opt_config['name']} and learning rate {lr}")

        # Generate hash for the current optimizer configuration
        opt_config_copy = copy.deepcopy(opt_config)
        opt_config_copy['lr'] = lr
        config_hash = hash_config(opt_config_copy, training_params, config['model_config'])
        file_name = f"{opt_config['name']}-ns{opt_config.get('ns_steps', 5)}-lr-{lr}-{opt_config['lr_schedule']}-{config_hash}-world{world_size}"
        if args.suffix != '': file_name += f"-{args.suffix}"
        output_path = os.path.join(output_dir, file_name + '.json')
        ckpt_dir = os.path.join(ckpt_dir_base, file_name) + '/' if CKPT_DIR != "" else ""

        # copy model to ensure consistency
        model_copy = copy.deepcopy(model).to(device)
        if training_params['compile']:
            if master_process: print("Compiling model")
            model_copy = torch.compile(model_copy)
        if ddp:
            model_copy = DDP(model_copy, device_ids=[local_rank])

        # Setup optimizer
        optimizer_obj, hyperp = get_optimizer(opt_config, lr=lr)
        params = model_copy.named_parameters() if optimizer_obj is Muon2 else model_copy.parameters()
        optimizer = optimizer_obj(params, **hyperp)
        scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)

        # Initialize wandb
        wandb_run = None
        if master_process and config['logging_params'].get('wandb', None) is not None:
            import wandb
            config_no_optimizer = copy.deepcopy(config)
            del config_no_optimizer['optimizer_params']
            wandb_config = dict(one_optimizer_params=opt_config_copy, **config_no_optimizer, world_size=world_size)
            wandb_run = wandb.init(
                **config['logging_params']['wandb'],
                config=wandb_config,
                name=args.prefix + file_name,
                reinit=True,
            )

        # Train
        try:
            logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                           scheduler=scheduler, ckpt_dir=ckpt_dir,
                           logging_params=config['logging_params'], wandb_run=wandb_run)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        # Save
        if master_process:
            logger.name = f"{opt_config['name']}-ns{opt_config.get('ns_steps', 5)}-lr-{lr}"
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting")
            with open(output_path, 'w') as file:
                json.dump(logger.__dict__, file)
            print(f"Saved output to {output_path}")

if ddp:
    dist.destroy_process_group()
