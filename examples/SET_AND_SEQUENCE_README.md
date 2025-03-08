# Set-and-Sequence: Dynamic Concepts Personalization

This implementation is based on the paper [Dynamic Concepts Personalization from Single Videos](https://snap-research.github.io/dynamic_concepts) by Abdal et al.

## Overview

Set-and-Sequence is a two-stage approach for personalizing text-to-video models with dynamic concepts - entities defined not only by their appearance but also by their unique motion patterns. The approach works by:

1. **Stage 1 (Identity Basis)**: Training on an unordered set of frames from a video to learn the appearance of the dynamic concept.
2. **Stage 2 (Motion Residual)**: Training on the full video sequence with the Identity Basis frozen to learn the motion dynamics.

This implementation supports the Wan 2.1 text-to-video model.

## Requirements

- A video file containing the dynamic concept you want to personalize
- Appropriate text prompts describing the appearance and motion
- GPU with sufficient VRAM (at least 24GB recommended)

## Usage

### Step 1: Prepare Your Configuration Files

1. Create a model configuration file based on `examples/set_and_sequence_config.toml`
2. Create a dataset configuration file based on `examples/set_and_sequence_dataset.toml`
3. Update the paths and settings in both files to match your environment and data

### Step 2: Run the Training

You have two options for running the Set-and-Sequence training:

#### Option 1: Run Both Stages Sequentially (Recommended)

Run both Stage 1 and Stage 2 in a single command:

```bash
python train_set_and_sequence.py \
  --config /path/to/your/config.toml \
  --run_both_stages \
  --stage1_epochs 600 \
  --stage2_epochs 900
```

This will:
1. Run Stage 1 (Identity Basis) for 600 epochs
2. Automatically use the Identity Basis from Stage 1 for Stage 2
3. Run Stage 2 (Motion Residual) for 900 epochs
4. Save all outputs in organized subdirectories

The final combined LoRA will be available at `/path/to/output/run/combined_lora/adapter_model.safetensors`.

#### Option 2: Run Each Stage Separately

If you prefer to run each stage separately, you can do so with the following commands:

**Stage 1 (Identity Basis)**:

```bash
python train_set_and_sequence.py \
  --config /path/to/your/config.toml \
  --stage 1
```

**Stage 2 (Motion Residual)**:

```bash
python train_set_and_sequence.py \
  --config /path/to/your/config.toml \
  --stage 2 \
  --identity_basis_path /path/to/stage1/output/run/identity_basis
```

### Additional Command-Line Options

- `--stage1_epochs`: Number of epochs for Stage 1 (overrides config)
- `--stage2_epochs`: Number of epochs for Stage 2 (overrides config)
- `--resume_from_checkpoint`: Resume training from the latest checkpoint
- `--regenerate_cache`: Force regenerate the dataset cache

## Configuration Options

### Set-and-Sequence Specific Options

The `set_and_sequence` section in the config file contains the following options:

- `stage1_dropout`: Dropout probability for B matrix in LoRA during Stage 1 (default: 0.8)
- `stage2_dropout`: Dropout probability for B matrix in LoRA during Stage 2 (default: 0.5)
- `text_token_mask_prob`: Probability of masking text tokens for regularization (default: 0.1)
- `self_conditioning_prob`: Probability of using self-conditioning (default: 0.9)
- `prior_preservation`: Whether to use prior preservation (default: true)
- `prior_preservation_weight`: Weight for prior preservation loss (default: 1.0)

## Tips for Best Results

1. **Video Quality**: Use high-quality videos with clear visibility of the dynamic concept.
2. **Video Length**: Videos should be at least a few seconds long to capture meaningful motion.
3. **Text Prompts**: Provide detailed prompts that describe both the appearance and motion of the concept.
4. **Training Duration**: 
   - Stage 1 typically requires 600-800 epochs
   - Stage 2 typically requires 900-1200 epochs for simple motions, and up to 2500 epochs for complex motions
5. **LoRA Rank**: Use a higher rank (32 or 64) for better capacity to capture both appearance and motion.

## Example Prompts

For a video of ocean waves:

- Stage 1 (Appearance): "A [v] ocean with blue water and white foam"
- Stage 2 (Motion): "A [v] ocean with blue water and white foam in [u] flowing motion"

For a video of a person dancing:

- Stage 1 (Appearance): "A [v] person wearing a red shirt and black pants"
- Stage 2 (Motion): "A [v] person wearing a red shirt and black pants performing [u] dance movements"

## Inference

After training both stages, you can use the final LoRA adapter with the Wan model for inference. The adapter will contain both the appearance and motion information of the dynamic concept.

### Using LoRAs in ComfyUI

The Set-and-Sequence implementation saves LoRAs in a ComfyUI-compatible format:

1. **Stage 1 (Identity Basis)**: Saved in `run/identity_basis/adapter_model.safetensors`
2. **Stage 2 (Combined)**: Saved in `run/combined_lora/adapter_model.safetensors`

To use these LoRAs in ComfyUI:

1. Copy the desired LoRA file to your ComfyUI's LoRA directory
2. In ComfyUI, load the Wan model and connect the LoRA node to it
3. Set the appropriate strength (typically 1.0 for full effect)

You can use the Identity Basis LoRA alone to capture just the appearance, or the combined LoRA to capture both appearance and motion.

### Example ComfyUI Workflow

Here's a basic workflow for using the Set-and-Sequence LoRAs in ComfyUI:

1. Load the Wan 2.1 model
2. Add a LoRA Loader node and connect it to the model
3. Set the LoRA path to your trained adapter
4. Add a prompt with your concept tokens (e.g., "A [v] ocean with waves in [u] motion")
5. Generate your video

## Limitations

- This implementation currently only supports the Wan 2.1 model
- Training requires significant GPU memory and time
- Complex motions may require longer training times 