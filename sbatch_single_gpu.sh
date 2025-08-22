#!/bin/sh
#SBATCH --partition=learn
#SBATCH --qos=genca_3drecon
#SBATCH --time=4-00:00:00
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-task=32
#SBATCH --error=/home/%u/logs/job.%J.err
#SBATCH --output=/home/%u/logs/job.%J.out

source /shared/conda_envs/latest-env/bin/activate
cd /home/$USER/rsc/UniAnimate-DiT/
export PYTHONPATH=.:$PYTHONPATH
echo "Start training"
python examples/unianimate_wan/train_unianimate_wan_sstk.py   --task train     --train_architecture lora    --lora_rank 64 --lora_alpha 64     --dataset_path data/example_dataset/SSTK350K/train_static_scenes.txt      --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk350k      \
    --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"       \
    --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    --max_epochs 10   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing \
    --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload \
    --dataloader_num_workers 16

python examples/unianimate_wan/train_unianimate_wan_sstk_ver1_ref.py   --task train     --train_architecture lora    --lora_rank 64 --lora_alpha 64     --dataset_path data/example_dataset      --output_path /gen_ca/j1wen/UniAnimate-DiT/models_out_sstk350k      \
    --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"       \
    --text_encoder_path "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"    --vae_path "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"    --max_epochs 10   --learning_rate 1e-4      --accumulate_grad_batches 1      --use_gradient_checkpointing \
    --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload \
    --dataloader_num_workers 16 --disable_ref_pose
