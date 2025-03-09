import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
import copy
from pathlib import Path

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np
import safetensors.torch

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.')
parser.add_argument('--cache_only', action='store_true', default=None, help='Cache model inputs then exit.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', default=None, help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--stage1_epochs', type=int, default=None, help="Number of epochs for Stage 1 (overrides config)")
parser.add_argument('--stage2_epochs', type=int, default=None, help="Number of epochs for Stage 2 (overrides config)")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Provide a default value for save_every_n_epochs instead of asserting its presence
    config.setdefault('save_every_n_epochs', 10)  # Default: save every 10 epochs
    
    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    # Set-and-Sequence specific defaults
    config.setdefault('set_and_sequence', {})
    sns_config = config['set_and_sequence']
    sns_config.setdefault('stage1_dropout', 0.8)
    sns_config.setdefault('stage2_dropout', 0.5)
    sns_config.setdefault('text_token_mask_prob', 0.1)
    sns_config.setdefault('self_conditioning_prob', 0.9)
    sns_config.setdefault('prior_preservation', True)
    sns_config.setdefault('prior_preservation_weight', 1.0)
    
    # Add separate epoch counts for each stage
    sns_config.setdefault('stage1_epochs', config['epochs'])
    sns_config.setdefault('stage2_epochs', config['epochs'])

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('enable_tensorboard', False)


def get_most_recent_run_dir(output_dir):
    run_dir = os.path.join(output_dir, "run")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir


def print_model_info(model):
    if not is_main_process():
        return
    
    print(f'Model type: {model.name}')
    
    # Check if diffusion model is loaded using a safer approach
    # Directly check the __dict__ instead of using hasattr which triggers __getattr__
    if 'diffusion_model' not in model.__dict__ or model.__dict__['diffusion_model'] is None:
        print("Diffusion model not loaded yet. Load the model before printing detailed parameters.")
        return
        
    # Now it's safe to access transformer
    total_params = 0
    trainable_params = 0
    for name, p in model.transformer.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%')


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.reset()
    eval_dataloader.set_timestep_quantile(quantile)
    eval_dataloader.set_epoch(0)
    eval_dataloader.sync_epoch()
    total_loss = 0
    num_batches = 0
    while True:
        model_engine.reset_activation_shape()
        loss = model_engine.eval_batch().item()
        total_loss += loss
        num_batches += 1
        eval_dataloader.sync_epoch()
        if pbar is not None:
            pbar.update(1)
        if eval_dataloader.epoch > 0:
            break
    return total_loss / num_batches


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    model_engine.eval()
    total_batches = 0
    for eval_dataloader in eval_dataloaders.values():
        total_batches += len(eval_dataloader) // eval_gradient_accumulation_steps
    total_batches *= len(TIMESTEP_QUANTILES_FOR_EVAL)
    with tqdm(total=total_batches, desc='Evaluating', disable=not is_main_process()) as pbar:
        for name, eval_dataloader in eval_dataloaders.items():
            for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
                loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar)
                if is_main_process() and tb_writer is not None:
                    tb_writer.add_scalar(f'eval/{name}_loss_q{quantile}', loss, step)
    model_engine.train()


def evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    if not eval_dataloaders:
        return
    with torch.no_grad():
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)


def apply_dropout_to_lora_b(model, dropout_prob):
    """Apply dropout to LoRA B matrices in a DeepSpeed PipelineModule"""
    # For DeepSpeed PipelineModule
    if hasattr(model, 'forward_funcs'):
        # Iterate through the layers in the pipeline
        for layer_idx, layer in enumerate(model.forward_funcs):
            # Check if the layer has modules
            if hasattr(layer, 'module'):
                # Apply dropout to all modules in the layer
                for name, module in layer.module.named_modules():
                    if hasattr(module, 'lora_dropout'):
                        # Apply dropout to LoRA B matrices
                        if dropout_prob > 0:
                            module.lora_dropout.p = dropout_prob
                        else:
                            module.lora_dropout.p = 0.0
    # For regular PyTorch modules
    elif hasattr(model, 'named_modules'):
        for name, module in model.named_modules():
            if hasattr(module, 'lora_dropout'):
                # Apply dropout to LoRA B matrices
                if dropout_prob > 0:
                    module.lora_dropout.p = dropout_prob
                else:
                    module.lora_dropout.p = 0.0
    else:
        if is_main_process():
            print(f"Warning: Model of type {type(model)} does not support LoRA dropout. Skipping.")


def mask_text_tokens(text_tokens, mask_prob):
    """Randomly mask text tokens with the given probability"""
    if mask_prob <= 0:
        return text_tokens
    
    # Create a copy to avoid modifying the original
    masked_tokens = copy.deepcopy(text_tokens)
    
    # For each sample in the batch
    for i in range(len(masked_tokens)):
        # Find non-padding tokens
        non_pad_indices = (masked_tokens[i] != 0).nonzero(as_tuple=True)[0]
        
        # Skip if there are no non-padding tokens
        if len(non_pad_indices) == 0:
            continue
        
        # Randomly select tokens to mask
        num_to_mask = max(1, int(len(non_pad_indices) * mask_prob))
        indices_to_mask = random.sample(non_pad_indices.tolist(), num_to_mask)
        
        # Mask selected tokens (replace with 0 or a special mask token)
        for idx in indices_to_mask:
            masked_tokens[i, idx] = 0  # Using 0 as mask token
    
    return masked_tokens


def train_stage1_identity_basis(model, config, train_data, eval_data_map, run_dir, resume_from_checkpoint):
    """Stage 1: Identity Basis Learning - Train on unordered set of frames"""
    if is_main_process():
        print("Starting Stage 1: Identity Basis Learning")
        print("Training on unordered set of frames to learn static identity")
    
    # Set high dropout for B matrix in LoRA
    dropout_prob = config['set_and_sequence']['stage1_dropout']
    
    # Train on static frames (no temporal information)
    # This is handled by the dataset configuration
    
    # Start training
    train_model(model, config, train_data, eval_data_map, run_dir, resume_from_checkpoint, 
                stage=1, dropout_prob=dropout_prob)
    
    # Save the Identity Basis LoRA in ComfyUI-compatible format
    if is_main_process():
        print("Saving Identity Basis LoRA in ComfyUI-compatible format")
        identity_basis_dir = Path(run_dir) / "identity_basis"
        os.makedirs(identity_basis_dir, exist_ok=True)
        
        # Get state dict of the LoRA
        state_dict = {}
        for name, p in model.transformer.named_parameters():
            if hasattr(p, 'original_name') and p.requires_grad:
                state_dict[p.original_name] = p.data.detach().cpu()
        
        # Save in ComfyUI format
        model.peft_config.save_pretrained(identity_basis_dir)
        
        # Format state dict according to model's convention (e.g., for Wan model, prefix with 'diffusion_model.')
        if model.name == 'wan':
            state_dict = {'diffusion_model.'+k: v for k, v in state_dict.items()}
        elif model.name == 'hunyuan-video':
            state_dict = {'transformer.'+k: v for k, v in state_dict.items()}
        
        # Save the adapter model
        safetensors.torch.save_file(state_dict, identity_basis_dir / 'identity_basis_adapter_model.safetensors', metadata={'format': 'pt'})
        
        # Also save a copy as adapter_model.safetensors for compatibility
        safetensors.torch.save_file(state_dict, identity_basis_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})
        
        print(f"Identity Basis LoRA saved to {identity_basis_dir}")


def train_stage2_motion_residual(model, config, train_data, eval_data_map, run_dir, resume_from_checkpoint, identity_basis_path):
    """Stage 2: Motion Residual Encoding - Train on full video sequence with frozen identity basis"""
    if is_main_process():
        print("Starting Stage 2: Motion Residual Encoding")
        print("Training on full video sequence to learn motion dynamics")
        print(f"Loading Identity Basis from: {identity_basis_path}")
    
    # Load the identity basis from Stage 1
    if not os.path.exists(identity_basis_path):
        raise ValueError(f"Identity Basis path {identity_basis_path} does not exist")
    
    # Set lower dropout for B matrix in LoRA for Stage 2
    dropout_prob = config['set_and_sequence']['stage2_dropout']
    
    # Load the identity basis and freeze it
    model.load_adapter_weights(identity_basis_path)
    
    # Freeze the A matrices (identity basis) and only train the B matrices (motion residuals)
    for name, param in model.transformer.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = False
    
    # Start training
    train_model(model, config, train_data, eval_data_map, run_dir, resume_from_checkpoint, 
                stage=2, dropout_prob=dropout_prob)
    
    # Save the final combined LoRA (Identity Basis + Motion Residual) in ComfyUI-compatible format
    if is_main_process():
        print("Saving combined Identity Basis + Motion Residual LoRA in ComfyUI-compatible format")
        combined_lora_dir = Path(run_dir) / "combined_lora"
        os.makedirs(combined_lora_dir, exist_ok=True)
        
        # Get state dict of the LoRA
        state_dict = {}
        for name, p in model.transformer.named_parameters():
            if hasattr(p, 'original_name'):
                state_dict[p.original_name] = p.data.detach().cpu()
        
        # Save in ComfyUI format
        model.peft_config.save_pretrained(combined_lora_dir)
        
        # Format state dict according to model's convention (e.g., for Wan model, prefix with 'diffusion_model.')
        if model.name == 'wan':
            state_dict = {'diffusion_model.'+k: v for k, v in state_dict.items()}
        elif model.name == 'hunyuan-video':
            state_dict = {'transformer.'+k: v for k, v in state_dict.items()}
        
        # Save the adapter model
        safetensors.torch.save_file(state_dict, combined_lora_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})
        
        print(f"Combined LoRA saved to {combined_lora_dir}")


def train_model(model, config, train_data, eval_data_map, run_dir, resume_from_checkpoint, stage=1, dropout_prob=0.0):
    """Common training function for both stages"""
    # Get optimizer parameters
    parameters = model.get_param_groups(model.transformer.parameters())
    
    # Configure optimizer
    optimizer_config = config['optimizer']
    optimizer_type = optimizer_config['type'].lower()
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps'],
        )
    elif optimizer_type == 'adamw_optimi':
        try:
            import optimi
            optimizer = optimi.AdamW(
                parameters,
                lr=optimizer_config['lr'],
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay'],
                eps=optimizer_config['eps'],
            )
        except ImportError:
            if is_main_process():
                print("optimi package not found. Falling back to standard AdamW.")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=optimizer_config['lr'],
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay'],
                eps=optimizer_config['eps'],
            )
    elif optimizer_type == 'adamw8bitkahan':
        try:
            from optimizers.adamw_8bit_kahan import AdamW8bitKahan
            optimizer = AdamW8bitKahan(
                parameters,
                lr=optimizer_config['lr'],
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay'],
                eps=optimizer_config['eps'],
                gradient_release=optimizer_config.get('gradient_release', False),
            )
        except ImportError:
            if is_main_process():
                print("AdamW8bitKahan not found. Falling back to standard AdamW.")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=optimizer_config['lr'],
                betas=optimizer_config['betas'],
                weight_decay=optimizer_config['weight_decay'],
                eps=optimizer_config['eps'],
            )
    else:
        if is_main_process():
            print(f"Optimizer type {optimizer_type} not specifically implemented. Falling back to standard AdamW.")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_config['lr'],
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            eps=optimizer_config.get('eps', 1e-8),
        )

    # Configure DeepSpeed
    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    if config['activation_checkpointing']:
        additional_pipeline_module_kwargs['activation_checkpoint_interval'] = 1
        additional_pipeline_module_kwargs['activation_checkpoint_func'] = isolate_rng(torch.utils.checkpoint.checkpoint)
    
    # Create pipeline model
    pipeline_model = deepspeed.pipe.PipelineModule(
        layers=layers,
        num_stages=config['pipeline_stages'],
        partition_method=config.get('partition_method', 'parameters'),
        loss_fn=lambda x, y: x,
        **additional_pipeline_module_kwargs
    )
    
    # Configure DeepSpeed engine
    ds_config = {
        'train_micro_batch_size_per_gpu': config['micro_batch_size_per_gpu'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'steps_per_print': config['steps_per_print'],
        'gradient_clipping': config['gradient_clipping'],
        'wall_clock_breakdown': False,
        'zero_optimization': {
            'stage': 0,
        },
        'pipeline': {
            'activation_checkpoint_interval': 0
        }
    }
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        optimizer=optimizer,
        config=ds_config
    )
    
    # Initialize data loaders
    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
        )
    
    # NOW we can create the learning rate scheduler after dataset is initialized
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_data) // config['gradient_accumulation_steps'] * config['epochs'],
    )
    
    # Apply warmup scheduler if configured
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler
    
    # Set communication data type
    communication_data_type = config['adapter']['dtype'] if 'adapter' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type
    
    # Create data loader
    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)
    
    # Resume from checkpoint if requested
    step = 1
    if resume_from_checkpoint:
        client_state = {}
        _, client_state = model_engine.load_checkpoint(run_dir, tag='latest')
        step = client_state['step'] + 1
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')
    
    # Set constant learning rate if configured
    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']
    
    # Set dataloader and calculate steps
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']
    
    # Create evaluation dataloaders
    eval_dataloaders = {
        # Set num_dataloader_workers=0 for deterministic validation
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }
    
    # Initialize tensorboard writer and saver
    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() and config['enable_tensorboard'] else None
    saver = utils.saver.Saver(args, config, True, run_dir, model, train_dataloader, model_engine, pipeline_model)
    
    # Evaluate before first step if configured
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'])
    
    # Initialize training state
    epoch_loss = 0
    num_steps = 0
    
    # Set-and-Sequence specific configurations
    text_token_mask_prob = config['set_and_sequence']['text_token_mask_prob']
    self_conditioning_prob = config['set_and_sequence']['self_conditioning_prob']
    
    # Main training loop
    while True:
        # Apply dropout to LoRA B matrices
        apply_dropout_to_lora_b(model_engine, dropout_prob)
        
        # Apply text token masking if configured
        if text_token_mask_prob > 0 and hasattr(train_dataloader, 'current_batch') and 'text_tokens' in train_dataloader.current_batch:
            train_dataloader.current_batch['text_tokens'] = mask_text_tokens(
                train_dataloader.current_batch['text_tokens'], 
                text_token_mask_prob
            )
        
        # Reset activation shape and train batch
        model_engine.reset_activation_shape()
        loss = model_engine.train_batch().item()
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()
        
        # Process epoch and save checkpoints
        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False
        
        # Log to tensorboard
        if is_main_process() and tb_writer is not None and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, step)
            tb_writer.add_scalar(f'train/stage', stage, step)
        
        # Evaluate if configured
        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'])
        
        # Log epoch loss
        if finished_epoch:
            if is_main_process() and tb_writer is not None:
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
            epoch_loss = 0
            num_steps = 0
            epoch = new_epoch
        
        # Check if training is complete
        if epoch >= config['epochs']:
            break
        
        step += 1
    
    # Final evaluation
    if config['eval_every_n_epochs'] or config['eval_every_n_steps']:
        evaluate(model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'])
    
    # Save final checkpoint
    if is_main_process():
        print('Training complete!')


def run_both_stages(config, train_data, eval_data_map, run_dir, resume_from_checkpoint):
    """Run both Stage 1 and Stage 2 sequentially"""
    if is_main_process():
        print("Running both Stage 1 and Stage 2 sequentially")
    
    # Load model
    model_type = config['model']['type']
    if model_type == 'wan':
        from models.wan import WanPipeline
        model = WanPipeline(config)
    else:
        raise ValueError(f'Model type {model_type} is not supported for Set-and-Sequence training')
    
    # Print basic model info before loading diffusion model
    if is_main_process():
        print(f'Model type: {model.name}')

    # Load diffusion model
    model.load_diffusion_model()

    # Configure adapter
    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
    else:
        raise ValueError('Set-and-Sequence requires a LoRA adapter configuration')

    # Now print detailed model info
    print_model_info(model)
    
    # Override epochs for Stage 1 if specified
    original_epochs = config['epochs']
    if args.stage1_epochs is not None:
        config['epochs'] = args.stage1_epochs
    elif 'stage1_epochs' in config['set_and_sequence']:
        config['epochs'] = config['set_and_sequence']['stage1_epochs']
    
    # Run Stage 1
    if is_main_process():
        print(f"Starting Stage 1 with {config['epochs']} epochs")
    
    # Create a subdirectory for Stage 1
    stage1_dir = os.path.join(run_dir, "stage1")
    if is_main_process():
        os.makedirs(stage1_dir, exist_ok=True)
        # Copy config file to stage1 directory
        if hasattr(args, 'config') and os.path.exists(args.config):
            shutil.copy(args.config, stage1_dir)
    
    # Wait for all processes
    dist.barrier()
    
    # For Stage 1, modify dataset config to use static frames
    dataset_config = toml.load(config['dataset'])
    for directory in dataset_config.get('directory', []):
        directory['static_frames'] = True
    
    # Create dataset manager and dataset for Stage 1
    caching_batch_size = config.get('caching_batch_size', 1)
    stage1_dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=args.regenerate_cache, caching_batch_size=caching_batch_size)
    stage1_dataset = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    stage1_dataset_manager.register(stage1_dataset)
    
    # Register eval datasets
    for name, eval_data in eval_data_map.items():
        stage1_dataset_manager.register(eval_data)
    
    # Cache datasets
    stage1_dataset_manager.cache()
    
    # Run Stage 1
    train_stage1_identity_basis(model, config, stage1_dataset, eval_data_map, stage1_dir, resume_from_checkpoint)
    
    # Path to the Identity Basis LoRA
    identity_basis_path = os.path.join(stage1_dir, "identity_basis")
    
    # Override epochs for Stage 2 if specified
    if args.stage2_epochs is not None:
        config['epochs'] = args.stage2_epochs
    elif 'stage2_epochs' in config['set_and_sequence']:
        config['epochs'] = config['set_and_sequence']['stage2_epochs']
    else:
        config['epochs'] = original_epochs
    
    # Create a new model instance for Stage 2
    if model_type == 'wan':
        from models.wan import WanPipeline
        model = WanPipeline(config['model'])
    
    # Print model info
    print_model_info(model)
    
    # Load diffusion model
    model.load_diffusion_model()
    
    # Configure adapter
    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
    
    # Create a subdirectory for Stage 2
    stage2_dir = os.path.join(run_dir, "stage2")
    if is_main_process():
        os.makedirs(stage2_dir, exist_ok=True)
        # Copy config file to stage2 directory
        if hasattr(args, 'config') and os.path.exists(args.config):
            shutil.copy(args.config, stage2_dir)
    
    # Wait for all processes
    dist.barrier()
    
    # For Stage 2, modify dataset config to use full video sequences
    dataset_config = toml.load(config['dataset'])
    for directory in dataset_config.get('directory', []):
        directory['static_frames'] = False
    
    # Create dataset manager and dataset for Stage 2
    stage2_dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=args.regenerate_cache, caching_batch_size=caching_batch_size)
    stage2_dataset = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    stage2_dataset_manager.register(stage2_dataset)
    
    # Register eval datasets
    for name, eval_data in eval_data_map.items():
        stage2_dataset_manager.register(eval_data)
    
    # Cache datasets
    stage2_dataset_manager.cache()
    
    # Run Stage 2
    if is_main_process():
        print(f"Starting Stage 2 with {config['epochs']} epochs")
        print(f"Using Identity Basis from: {identity_basis_path}")
    
    train_stage2_motion_residual(model, config, stage2_dataset, eval_data_map, stage2_dir, False, identity_basis_path)
    
    # Create a symbolic link to the final combined LoRA in the run directory
    if is_main_process():
        combined_lora_dir = os.path.join(stage2_dir, "combined_lora")
        final_lora_dir = os.path.join(run_dir, "combined_lora")
        
        if os.path.exists(combined_lora_dir):
            # Create a symbolic link or copy the files
            if not os.path.exists(final_lora_dir):
                try:
                    os.symlink(combined_lora_dir, final_lora_dir)
                except OSError:
                    # If symlink fails (e.g., on Windows), copy the files
                    shutil.copytree(combined_lora_dir, final_lora_dir)
            
            print(f"Final combined LoRA available at: {final_lora_dir}")
    
    if is_main_process():
        print("Completed both stages of Set-and-Sequence training!")


if __name__ == '__main__':
    # Apply patches
    apply_patches()
    
    # Load config
    if args.config is None:
        raise ValueError('--config is required')
    
    if not os.path.exists(args.config):
        raise ValueError(f'Config file {args.config} does not exist')
    
    config = toml.load(args.config)
    set_config_defaults(config)
    
    # Initialize DeepSpeed before caching
    deepspeed.init_distributed()
    # Set device for distributed training
    torch.cuda.set_device(dist.get_rank())
    
    # Create output directory
    output_dir = config.get('output_dir', 'output')
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # Create run directory
    run_dir = None
    if is_main_process():
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(run_dir, 'config.toml'), 'w') as f:
            toml.dump(config, f)
    
    # Broadcast run_dir to all processes
    if dist.get_world_size() > 1:
        run_dir_list = [run_dir]
        torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]
    
    # Load model
    model_type = config['model']['type']
    if model_type == 'wan':
        from models.wan import WanPipeline
        model = WanPipeline(config)
    else:
        raise ValueError(f'Model type {model_type} is not supported for Set-and-Sequence training')
    
    # Print basic model info before loading diffusion model
    if is_main_process():
        print(f'Model type: {model.name}')

    # Load diffusion model
    model.load_diffusion_model()

    # Configure adapter
    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
    else:
        raise ValueError('Set-and-Sequence requires a LoRA adapter configuration')

    # Now print detailed model info
    print_model_info(model)
    
    # Load dataset
    dataset_config = toml.load(config['dataset'])
    
    # For model-specific validation
    model.model_specific_dataset_config_validation(dataset_config)
    
    # Create dataset manager
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=args.regenerate_cache, caching_batch_size=caching_batch_size)
    
    # Create dataset
    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)
    
    # Create evaluation datasets
    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])
    
    # Cache datasets
    dataset_manager.cache()
    
    # Exit if only caching
    if args.cache_only:
        quit()
    
    # Run both stages sequentially
    run_both_stages(config, train_data, eval_data_map, run_dir, args.resume_from_checkpoint) 