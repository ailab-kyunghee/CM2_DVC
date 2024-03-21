conda activate cm2

#config_name and eval_folder must be same for tracking save folder. 
#exp_name will be added with eval_folder (./save/eval_folder+'_'+exp_name)
exp_name=Clip_anet_100k_20window_30ep
config_path=cfgs/anet_clip_cm2.yml
eval_folder=anet_clip_cm2

#Training
python train.py --text_crossAttn --ret_encoder avg --cfg_path ${config_path} --exp_name ${exp_name} --sim_attention cls_token --target_domain anet --bank_type anet --soft_k 100 --window_size 20  #--retrieval_ablation no_ret  #--nvec_proj_use True #--soft_k 25 

# Evaluation
python eval.py --text_crossAttn --target_domain anet --bank_type anet --text_crossAttn_loc after --sim_match window_cos --ret_text token --down_proj deep --eval_folder ${eval_folder} --exp_name ${exp_name} --eval_transformer_input_type queries --able_ret --sim_attention cls_token  #--nvec_proj_use #--soft_k 25    #--proj_use 