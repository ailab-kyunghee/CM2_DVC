############################################################################# yc2
conda activate cm2
exp_name=best
config_path=cfgs/yc2_clip_cm2.yml
eval_folder=yc2_clip_cm2
python eval.py --bank_type yc2 --text_crossAttn --text_crossAttn_loc after --sim_match window_cos --ret_text token --target_domain yc2 --down_proj deep --eval_folder ${eval_folder} --exp_name ${exp_name} --eval_transformer_input_type queries --able_ret --sim_attention cls_token 