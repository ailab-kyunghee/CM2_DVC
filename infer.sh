#!/bin/bash

#SBATCH --job-name=onlyCross30ep_inference_DVC_gpt28_4k_PDVCproposal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=30G
#SBATCH -w augi4
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o Out_folder/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/minkuk/init.sh
conda activate memCap

python infer.py --decoder_name gpt2 --bank_type anet --cap_task dense --backbone tsp --predicted_proposal PDVC --k 4 --model_path /data/minkuk/caption/experiments/09-04_00-34_finetune:False_rag_28.0M_gpt2_30epochs --checkpoint_path checkpoint-17550
# python infer.py --decoder_name gpt2 --bank_type anet --cap_task dense --backbone tsp --k 1 --bert_score --predicted_proposal --model_path /data/minkuk/caption/experiments/09-04_06-05_finetune:False_rag_28.0M_gpt2_30epochs --checkpoint_path checkpoint-17550

#gpt-medium
# python infer.py --decoder_name gpt2-medium --bank_type anet --cap_task dense --backbone tsp --predicted_proposal --model_path /data/minkuk/caption/experiments/09-04_12-44_finetune:False_rag_28.0M_gpt2-medium_50epochs --checkpoint_path checkpoint-8790