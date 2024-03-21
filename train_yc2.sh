conda activate cm2

#config_name and eval_folder must be same for tracking save folder. 
#exp_name will be added with eval_folder (./save/eval_folder+'_'+exp_name)

exp_name=Clip_yc2_80k_50window_30ep
config_path=cfgs/yc2_clip_cm2.yml
eval_folder=yc2_clip_cm2

# Training
python train.py --text_crossAttn_loc after --text_crossAttn --ret_encoder avg --cfg_path ${config_path} --exp_name ${exp_name} --sim_attention cls_token --target_domain yc2 --bank_type yc2 --soft_k 80 --window_size 50 #--retrieval_ablation no_ret  #--nvec_proj_use True #--soft_k 25 

# Evaluation
python eval.py --text_crossAttn_loc after --target_domain yc2 --bank_type yc2 --sim_match window_cos --ret_text token --down_proj deep --ret_vector nvec --eval_folder ${eval_folder} --exp_name ${exp_name} --eval_transformer_input_type queries --able_ret --sim_attention cls_token  #--nvec_proj_use #--soft_k 25    #--proj_use 