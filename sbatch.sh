#!/bin/sh
#SBATCH --partition=learn
#SBATCH --qos=encoders_shared
#SBATCH --time=7-00:00:00
#SBATCH --job-name=multi_gpu_zeroinit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=200G
#SBATCH --error=/home/%u/logs/job.%J.err
#SBATCH --output=/home/%u/logs/job.%J.out

source /shared/conda_envs/latest-env/bin/activate
cd /home/$USER/rsc/UniAnimate-DiT/
export PYTHONPATH=.:$PYTHONPATH
echo "Start training"
# python examples/unianimate_wan/train_unianimate_wan_sstk.py   --task train     --train_architecture lora    --lora_rank 64 --lora_alpha 64     --dataset_path data/example_dataset      --output_path ./models_out_one_GPU      --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"       --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    --max_epochs 10   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload

# python examples/unianimate_wan/train_unianimate_wan_sstk.py     \
#     --task train   --train_architecture lora    --lora_rank 128 --lora_alpha 128     --dataset_path data/example_dataset      \
#     --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk_8gpu   \
#     --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"        \
#     --max_epochs 50   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing    \
#     --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"    \
#     --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    \
#     --use_gradient_checkpointing_offload    --training_strategy "deepspeed_stage_2" \
#     --dataloader_num_workers 4

# python examples/unianimate_wan/train_unianimate_wan_sstk.py     \
#     --task train   --train_architecture lora    --lora_rank 128 --lora_alpha 128     --dataset_path data/example_dataset/SSTK350K/train_static_scenes.txt      \
#     --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk_static_8gpu   \
#     --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"        \
#     --max_epochs 50   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing    \
#     --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"    \
#     --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    \
#     --use_gradient_checkpointing_offload    --training_strategy "deepspeed_stage_2" \
#     --dataloader_num_workers 4

python examples/unianimate_wan/train_unianimate_wan_sstk.py     \
    --task train   --train_architecture lora    --lora_rank 128 --lora_alpha 128         \
    --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk_zeroinit_8gpu   \
    --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"        \
    --max_epochs 50   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing    \
    --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"    \
    --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    \
    --use_gradient_checkpointing_offload    --training_strategy "deepspeed_stage_2" \
    --dataloader_num_workers 4 --zero_init

# python examples/unianimate_wan/train_unianimate_wan_sstk_ver1_ref.py     \
#     --task train   --train_architecture lora    --lora_rank 128 --lora_alpha 128     --dataset_path data/example_dataset      \
#     --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk_ver1_ref_norefpose_8gpu   \
#     --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"        \
#     --max_epochs 50   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing    \
#     --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"    \
#     --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    \
#     --use_gradient_checkpointing_offload    --training_strategy "deepspeed_stage_2" \
#     --dataloader_num_workers 4 --disable_ref_pose
