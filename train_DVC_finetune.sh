#!/bin/bash

#SBATCH --job-name=DVC_28_LinearT_prompt_GPT2Large_1e4_30epoch
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH -w augi4
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o Out_folder/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/minkuk/init.sh
conda activate memCap

#gpt2base
# python train.py --attention_size 28 --num_gpu 2 --batch_size 32 --n_epochs 30 --backbone tsp --bank_type anet --lr 1e-4
#gpt2medium
python train.py --attention_size 28 --num_gpu 2 --batch_size 16 --n_epochs 50 --k 4 --backbone tsp --bank_type anet --lr 1e-4 --gradient_steps 2 --decoder_name gpt2-large