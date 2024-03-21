import argparse
import time
import yaml
import os
import numpy as np

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--able_ret",type=bool, default=True,help="if true, retrieval")
    
    parser.add_argument('--sim_match', type=str, default='window_cos',help='cosine,window_cos,... ')
    parser.add_argument('--sim_attention', type=str, default='cls_token',help='avg, cls_token, ... ')
    
    parser.add_argument("--proj_use",type=bool, default=False,help="if true, downProj")
    parser.add_argument("--nvec_proj_use",type=bool, default=False,help="if true,new vetor that made after soft attention will be downProj")
    parser.add_argument('--memory_type', type=str, default='clip',help='clip,video')
    parser.add_argument('--exp_name', type=str,help='save_path, cfgname + exp_name')
    parser.add_argument('--ret_ablation', type=str,default="None",help='ablation:Measure retrieval performance for upper bound / cheating_ablation : Measure cheated retrieval performance for upper bound ')
    parser.add_argument("--ideal_test",action="store_true", default=False,help="if true,ideal_test")
    parser.add_argument('--eval_mode', type=str, default=False, choices=['eval', 'test'])
    
    parser.add_argument('--caption_decoder_new', type=str, default="none",help="none,gpt2")
    
    parser.add_argument('--decoder_query_ablation', type=bool, default=True,help='True-on / False-off')
    parser.add_argument("--decoder_query_num", type=int,default=100)
    
    
    ################################
    # configure of this run
    parser.add_argument('--cfg_path', type=str, help='config file',default='cfgs/yc2_clip_cm2.yml')
    
    parser.add_argument("--soft_k", type=int,default=20)
    parser.add_argument("--target_domain", type=str, default='yc2', help="which domain will be used  // 'anet','yc2','image' ")
    parser.add_argument("--bank_type", nargs='+', default=['yc2'], help="which domain will be used in ret bank // ['anet','yc2','image']")
    parser.add_argument('--window_type', type=str, default='FHQ',help='FHQ,FH,FQ,HQ,Q,H,F')
    parser.add_argument('--window_size', type=int, default=80,help='window number')
    parser.add_argument("--retrieval_ablation", type=str , default='None', help="ideal,no_ret,none")
    parser.add_argument("--ret_encoder", type=str , default='attention', help="avg,miniTE,top1,attention")
    
    
    
    parser.add_argument('--window_avg_pooling', type=str, default='off',help='on / off')
    parser.add_argument('--text_transformer',action="store_true",default=False, help="Text transformer encoder with no weight sharing")
    parser.add_argument('--text_crossAttn', action="store_true", default=False,help='True-on / False-off')
    parser.add_argument('--text_crossAttn_loc', type=str, default='after',help='before / after')
    parser.add_argument('--combined_encoder', action="store_true", default=False, help='True-concat and encode / False-seperate and encode each')
    
    #######################################
    #for new struc    
    parser.add_argument('--ret_vector', type=str, default='nvec',help='nvec,memvec')
    # parser.add_argument('--ret_encdec', type=str, default=False,help='nvec,memvec')
    parser.add_argument('--ret_encdec_ref_num', type=int, default=5,help='default 10 / num of reference when using memory vector retrieval')
    parser.add_argument('--down_proj', type=str, default='deep',help='simple,deep')
    parser.add_argument('--ret_text', type=str, default='token',help='token,sentence// token level or sentence level retireval')  
    parser.add_argument('--ret_token_encoder', type=str, default="off",help='on/off - token level embedding encoding')  

    #for new loss
    parser.add_argument('--ret_loss', type=bool, default=False,help='ret loss')  
    parser.add_argument('--ret_loss_coef', default=1, type=float)

    parser.add_argument('--id', type=str, default='', help='id of this run. Results and logs will saved in this folder ./save/id')
    parser.add_argument('--gpu_id', type=str, nargs='+', default=[0,1])
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--random_seed',  action='store_true', help='choose a random seed from {1,...,1000}')
    parser.add_argument('--disable_cudnn', type=int, default=0, help='disable cudnn may solve some unknown bugs')
    parser.add_argument('--debug', action='store_true', help='using mini-dataset for fast debugging')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='device to use for training / testing')

    #  ***************************** INPUT DATA PATH *****************************
    parser.add_argument('--train_caption_file', type=str,
                        default='data/anet/captiondata/train_modified.json', help='')
    parser.add_argument('--invalid_video_json', type=str, nargs='+', default=[])
    parser.add_argument('--val_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--visual_feature_folder', type=str, default='data/anet/')
    parser.add_argument('--gt_file_for_auc', type=str, nargs='+', default='data/anet/captiondata/val_all.json')
    parser.add_argument('--gt_file_for_eval', type=str, nargs='+', default=['data/anet/captiondata/val_1.json', 'data/anet/captiondata/val_2.json'])
    parser.add_argument('--gt_file_for_para_eval', type=str, nargs='+', default= ['data/anet/captiondata/para/anet_entities_val_1_para.json', 'data/anet/captiondata/para/anet_entities_val_2_para.json'])
    parser.add_argument('--dict_file', type=str, default='data/anet/vocabulary_activitynet.json', help='')
    parser.add_argument('--criteria_for_best_ckpt', type=str, default='dvc', choices=['dvc', 'pc', 'grounding'], help='for dense video captioning, use soda_c + METEOR as the criteria'
                                                                                                         'for paragraph captioning, choose the best para_METEOR+para_CIDEr+para_BLEU4'
                                                                                                         'for temporal visual grounding, choose the best IOU0.3 + IOU0.5 + IOU0.7')

    parser.add_argument('--visual_feature_type', type=str, default='tsp', choices=['c3d', 'resnet_bn', 'resnet'])
    parser.add_argument('--feature_dim', type=int, default=500, help='dim of frame-level feature vector')

    parser.add_argument('--start_from', type=str, default='', help='id of the run with incompleted training')
    parser.add_argument('--start_from_mode', type=str, choices=['best', 'last'], default="last")
    parser.add_argument('--pretrain', type=str, choices=['full', 'encoder', 'decoder'])
    parser.add_argument('--pretrain_path', type=str, default='', help='path of .pth')

    #  ***************************** DATALOADER OPTION *****************************
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--data_norm', type=int, default=0)
    parser.add_argument('--data_rescale', type=int, default=1)

    parser.add_argument('--feature_sample_rate', type=int, default=1)
    parser.add_argument('--train_proposal_sample_num', type=int,
                        default=24,
                        help='number of sampled proposals (or proposal sequence), a bigger value may be better')
    parser.add_argument('--gt_proposal_sample_num', type=int, default=10)
    # parser.add_argument('--train_proposal_type', type=str, default='', choices=['gt', 'learnt_seq', 'learnt'])
    
    
    ############################ Retrieval ########################################
    parser.add_argument('--save_mode', type=bool, default=False,help='If true, using save mode')
    parser.add_argument('--save_path', type=str, default='-')
    parser.add_argument("--sentence_encoder",type=str, default='Trainable',help="if Trainable,custom. else, pretrained")
    parser.add_argument("--disable_ret_prefix",action="store_true", default=False,help="if true, no text prefix retrieval and no prefix")
    parser.add_argument("--bank_path",type = str, default = 'bank')
    # parser.add_argument("--k", type=int, default=1, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--parag_template_path", type=str, default="src/template.txt", help="TXT file with template")
    parser.add_argument("--dvc_template_path", type=str, default="src/dvc_template.txt", help="TXT file with template")
    parser.add_argument("--generator", type=str, default="base", help="base, gpt2, opt // Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--ablation_mode", type=str, default="base", help="ideal_test, ...")
    parser.add_argument("--sentence_embedder", type=str, default="t5", help="t5,t5_large, bert_large,")

    ###############################################################################

    #  ***************************** Caption Decoder  *****************************
    parser.add_argument('--vocab_size', type=int, default=5747)
    parser.add_argument('--wordRNN_input_feats_type', type=str, default='C', choices=['C', 'E', 'C+E'],
                        help='C:clip-level features, E: event-level features, C+E: both')
    parser.add_argument('--caption_decoder_type', type=str, default="light",
                        choices=['none','light', 'standard'])
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary')
    parser.add_argument('--att_hid_size', type=int, default=512, help='the hidden size of the attention MLP')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--max_caption_len', type=int, default=30, help='')

    #  ***************************** Transformer  *****************************
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--caption_cost_type', type=str, default='loss')
    parser.add_argument('--set_cost_caption', type=float, default=0)
    parser.add_argument('--set_cost_class', type=float, default=1)
    parser.add_argument('--set_cost_bbox', type=float, default=5)
    parser.add_argument('--set_cost_giou', type=float, default=2)
    parser.add_argument('--cost_alpha', type=float, default=0.25)
    parser.add_argument('--cost_gamma', type=float, default=2)

    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--count_loss_coef', default=0, type=float)
    parser.add_argument('--caption_loss_coef', default=0, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--transformer_ff_dim', type=int, default=2048)
    parser.add_argument('--transformer_dropout_prob', type=float, default=0.1)
    parser.add_argument('--frame_embedding_num', type=int, default = 100)
    parser.add_argument('--sample_method', type=str, default = 'nearest', choices=['nearest', 'linear'])
    parser.add_argument('--fix_xcw', type=int, default=0)


    #  ***************************** OPTIMIZER *****************************
    parser.add_argument('--training_scheme', type=str, default='all', choices=['cap_head_only', 'no_cap_head', 'all'])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--batch_size_for_eval', type=int, default=1, help='')
    parser.add_argument('--grad_clip', type=float, default=100., help='clip gradients at this value')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    parser.add_argument('--lr', type=float, default=1e-4, help='1e-4 for resnet feature and 5e-5 for C3D feature')
    parser.add_argument('--learning_rate_decay_start', type=float, default=8)
    parser.add_argument('--learning_rate_decay_every', type=float, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)

    #  ***************************** SAVING AND LOGGING *****************************
    parser.add_argument('--min_epoch_when_save', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=1)
    parser.add_argument('--save_all_checkpoint', action='store_true')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')

    #  ***************************** For Deformable DETR *************************************
    parser.add_argument('--lr_backbone_names', default=["None"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_proj', default=0, type=int)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--transformer_input_type', default='queries', choices=['gt_proposals', 'learnt_proposals', 'queries'])

    # * Backbone
    parser.add_argument('--backbone', default=None, type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer

    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--share_caption_head', type = int ,default=1)

    parser.add_argument('--cap_nheads', default=8, type=int)
    parser.add_argument('--cap_dec_n_points', default=4, type=int)
    parser.add_argument('--cap_num_feature_levels', default=4, type=int)
    parser.add_argument('--disable_mid_caption_heads', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")


    # * Loss coefficients

    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2., type=float)


    #***************************** Event counter *****************************
    parser.add_argument('--max_eseq_length', default=10, type=int)
    parser.add_argument('--lloss_gau_mask', default=1, type=int)
    parser.add_argument('--lloss_beta', default=1, type=float)

    # scheduled sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--basic_ss_prob', type=float, default=0, help='initial ss prob')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=2,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    
    
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    #################### Contrastive learning ######################
    parser.add_argument('--box_head_init_bias', type=float, default=-2.0)
    parser.add_argument('--task_heads_lr', type=float, default=5e-5)
    parser.add_argument('--task_heads_different_lr', action='store_true')

    parser.add_argument('--learning_strategy', type=str, default='multi_step',choices=('warmup_linear', 'multi_step', 'warmup_cosine'))
    parser.add_argument('--warm_up_ratio', type=float, default=0.1, help='Fraction of total number of steps')


    #  ***************************** OPTIMIZER *****************************
    parser.add_argument('--eval_batch_size', type=int, default=1, help='')

    parser.add_argument('--enable_pos_emb_for_captioner', action='store_true', default=False) ## added by 0_wt on 2022/03/01
    parser.add_argument('--caption_loss_type', type=str, default='ce')
    
    parser.add_argument('--train_use_amp', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='anet')


    #  ***************************** Text Encoder *****************************
    parser.add_argument('--pretrained_language_model', type=str, default='roberta-base',    help='Pretrained hugging face model')
    parser.add_argument('--load_pretrained_language_model_from_config', type=str, default=None, help='creating a randomly initialized model')
    parser.add_argument('--gpt_model', type=str, default='gpt2')
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5, help='Learning rate of text encoder')
    parser.add_argument('--text_encoder_learning_strategy', type=str, default='warmup_linear',choices=('warmup_linear', 'multi_step', 'frozen', 'warmup_cosine'))
    parser.add_argument('--text_encoder_warm_up_ratio', type=float, default=0.01, help='Fraction of total number of steps')
    parser.add_argument('--text_encoder_lr_decay_start', type=float, default=8)
    parser.add_argument('--text_encoder_lr_decay_every', type=float, default=3)
    parser.add_argument('--text_encoder_lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--max_text_input_len', type=int, default=32, help='')
    parser.add_argument('--enable_layer_diff_text_feature', type=bool, default=False,help='Aux layer will have different text feature from final layer if true')
    parser.add_argument('--enable_word_context_modeling', type=bool, default=False, help='')
    parser.add_argument('--word_context_modeling_type',  type=str, default='attention_pool')
    parser.add_argument('--enable_sentence_context_modeling', type=bool, default=False, help='If add extra self attention layer after text encoder')
    parser.add_argument('--enable_sentence_pos_embedding', type=bool, default=False)
    parser.add_argument('--sentence_pos_embedding_type', type=str, default='cosine')
    parser.add_argument('--enable_multilayer_projection', default=False)
    parser.add_argument('--max_pos_num', type=int, default=500)
    parser.add_argument('--sentence_modeling_layer_num', type=int, default=1)

    parser.add_argument('--enable_cross_model_fusion', type=bool, default=False)
    # proposal level attention loss
    parser.add_argument('--huggingface_cache_dir', type=str, default='.cache')

    #  ***************************** Contrastive Loss  *****************************
    parser.add_argument('--enable_contrastive', action='store_true', help='whether to use query-text contrastive loss')
    parser.add_argument('--contrastive_hidden_size', type=int, default=128, help='Contrastive hidden size')
    parser.add_argument('--contrastive_loss_start_coef', type=float, default=0.0, help='Weight of contrastive loss')
    parser.add_argument('--contrastive_loss_temperature', type=float, default=0.1, help='Temperature of cl temperature')
    parser.add_argument('--enable_cross_video_cl', type=bool, default=True, help='Enable cross video contrastive loss')
    parser.add_argument('--set_cost_cl', type=float, default=0.0)
    parser.add_argument('--cl_schedule_val', type=float, nargs='+', default=[0, 0.1])
    parser.add_argument('--cl_schedule_time', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--disable_cl_proj_layer_share_weight', action='store_true', help='use unshared weights for cl project layers')
    parser.add_argument('--enable_e2t_cl', action='store_true', help=' enable event-to-text contrastive')
    parser.add_argument('--enable_bg_for_cl', action='store_true', help=' add a class for background events')

    # finetuning captioner
    parser.add_argument('--only_ft_captioner', action='store_true', help='finetuning caption head needs loading pretrained weights')
    parser.add_argument('--ft_captioner_from_scratch', action='store_true', help='finetuning caption head without loading captioner weights')

    # finetune class caption head
    parser.add_argument('--only_ft_class_head', action='store_true', help='Linear probing for action detection')
    parser.add_argument('--action_classes_path', type=str, default='data/anet/anet1.3/action_name.txt')
    parser.add_argument('--tal_gt_file', type=str, default='data/anet/anet1.3/activity_net.v1-3.min.json')
    parser.add_argument('--support_mlp_class_head', action='store_true')

    # For grounding
    parser.add_argument('--eval_enable_grounding', default=True)
    parser.add_argument('--eval_enable_maximum_matching_for_grounding', default=False)
    parser.add_argument('--eval_set_cost_class', type=float, default=0.)
    parser.add_argument('--eval_grounding_cost_alpha', type=float, default=0.25)
    parser.add_argument('--eval_grounding_cost_gamma', type=float, default=2)
    parser.add_argument('--eval_set_cost_cl', type=float, default=1.0)
    parser.add_argument('--eval_disable_captioning', action='store_true', default=False)
    parser.add_argument('--eval_disable_contrastive', action='store_true', default=False)
    parser.add_argument('--eval_enable_matching_score', action='store_true', default=False)
    parser.add_argument('--eval_matching_score_weight', type=float, default=0.0)
    parser.add_argument('--eval_gt_file_for_grounding', type=str, default='data/anet/captiondata/grounding/val1_for_grounding.json')

    # Multi sentence grounding
    parser.add_argument('--train_with_split_anno', type=bool, default=False)

    # For fast evaluation
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021', '2018_cider'])
    
    # video cropping
    parser.add_argument('--enable_video_cropping', action='store_true', default=False)
    parser.add_argument('--min_crop_ratio', type=float, default=0.5)
    parser.add_argument('--crop_num', type=int, default=2)

    # reranking
    parser.add_argument('--ec_alpha', type=float, default=0.3)

    # GPT2 decode
    parser.add_argument('--prefix_num_mapping_layer', type=int, default=8)
    parser.add_argument('--prefix_size', type=int, default=512)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--eval_use_amp', action='store_true', default=False)

    # RL
    parser.add_argument('--rl_scorer_types', type=str, nargs='+', default=['Meteor'], choices=['Meteor', 'CiderD'])
    parser.add_argument('--rl_scorer_weights', type=float, nargs='+', default=[1.])
    parser.add_argument('--cached_tokens', type=str, default='anet/activitynet_train_ngrams_for_cider-idxs')
    parser.add_argument('--cl_para_ratio', type=float, default=0.0)
    parser.add_argument('--cl_sent_ratio', type=float, default=1.0)
    
    # reranking
    args = parser.parse_args()

    if args.cfg_path:
        import_cfg(args.cfg_path, vars(args))

    if args.random_seed:
        import random
        seed = int(random.random() * 1000)
        new_id = args.exp_name + '_seed{}'.format(seed)
        save_folder = os.path.join(args.save_dir, new_id)
        while os.path.exists(save_folder):
            seed = int(random.random() * 1000)
            new_id = args.exp_name + '_seed{}'.format(seed)
            save_folder = os.path.join(args.save_dir, new_id)
        args.id = new_id
        args.seed = seed

    if args.debug:
        args.id = 'debug_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.save_checkpoint_every = 1
        args.shuffle = 0

    if args.caption_decoder_type == 'none':
        assert args.caption_loss_coef == 0
        assert args.set_cost_caption == 0

    print("args.id: {}".format(args.id))
    export_to_json(args)
    return args

def import_cfg(cfg_path, args):
    with open(cfg_path, 'r') as handle:
        yml = yaml.load(handle, Loader=yaml.FullLoader)
        if 'base_cfg_path' in yml:
            base_cfg_path = yml['base_cfg_path']
            import_cfg(base_cfg_path, args)
        args.update(yml)
    pass

def export_to_json(args):
    # save a copy of all args in the lastest version,
    # used to recover the missing args when evaluating old runs by eval.py
    import json
    if not os.path.exists('.tmp'):
        os.mkdir('.tmp')
    json.dump(vars(args), open(".tmp/opts.json", 'w'))

if __name__ == '__main__':
    opt = parse_opts()
    export_to_json(opt)
    print(opt)