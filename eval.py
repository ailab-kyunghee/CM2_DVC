from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import torch
import numpy as np
import time
from os.path import dirname, abspath
import random
cm2_dir = dirname(abspath(__file__))
sys.path.insert(0, cm2_dir)
sys.path.insert(0, os.path.join(cm2_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(cm2_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from eval_utils_clip import evaluate
from cm2.cm2_ret_encdec_clip import build
from misc.utils import create_logger
from data.video_dataset import PropSeqDataset, collate_fn
from torch.utils.data import DataLoader
from os.path import basename
from ret_utils import load_clip_memory_bank
import pandas as pd
from misc.utils import set_seed
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
def create_fake_test_caption_file(metadata_csv_path):
    out = {}
    df = pd.read_csv(metadata_csv_path)
    for i, row in df.iterrows():
        out[basename(row['filename']).split('.')[0]] = {'duration': row['video-duration'], "timestamps": [[0, 0.5]], "sentences":["None"]}
    fake_test_json = '.fake_test_json.tmp'
    json.dump(out, open(fake_test_json, 'w'))
    return fake_test_json

def main(opt):
    folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    if opt.eval_mode == 'test':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    logger = create_logger(folder_path, 'val.log')
    if opt.eval_model_path:
        model_path = opt.eval_model_path
        infos_path = os.path.join('/'.join(opt.eval_model_path.split('/')[:-1]), 'info.json')
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')
        infos_path = os.path.join(folder_path, 'info.json')

    logger.info(vars(opt))

    print(print("Opt detail before load trained option ", opt))
    eval_retrieval_ablation=opt.retrieval_ablation
    eval_bank_type = opt.bank_type
    
    
    
    with open(infos_path, 'rb') as f:
        logger.info('load info from {}'.format(infos_path))
        old_opt = json.load(f)['best']['opt']

    for k, v in old_opt.items():
        if k[:4] != 'eval':
            vars(opt).update({k: v})
    
    
    print(print("Opt detail after load trained option ", opt))
    opt.retrieval_ablation=eval_retrieval_ablation
    opt.bank_type = eval_bank_type
    
    if opt.eval_gt_file_for_caption is not None:
        opt.gt_file_for_eval = opt.eval_gt_file_for_caption
    if True:
        # recover the lastest args
        if os.path.exists('.tmp/opts.json'):
            current_full_args = json.load(open('.tmp/opts.json'))
            for k,v in current_full_args.items():
                if k not in vars(opt):
                    vars(opt).update({k:v})
                    print('add missing args: {}={}'.format(k,v))
    opt.transformer_input_type = opt.eval_transformer_input_type
    opt.disable_tqdm = False
    opt.enable_init_query_embed = False
    opt.batch_size = opt.eval_batch_size

    if opt.eval_ec_alpha != -1:
        opt.ec_alpha = opt.eval_ec_alpha
    opt.enable_contrastive = False
    
    if opt.eval_disable_contrastive and opt.enable_contrastive:
        strict_load_pth = False
        opt.enable_contrastive = False
    elif opt.eval_not_strict_load:
        strict_load_pth = False
    else:
        strict_load_pth = True

    if not torch.cuda.is_available():
        opt.nthreads = 0
    # Create the Data Loader instance
    set_seed(opt.seed)
    val_dataset = PropSeqDataset(opt.eval_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, opt.eval_proposal_type,
                                 opt)
    loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                        shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    model, criterion, contrastive_criterion, postprocessors = build(opt)
    model.translator = val_dataset.translator

    while not os.path.exists(model_path):
        raise AssertionError('File {} does not exist'.format(model_path))

    logger.debug('Loading model from {}'.format(model_path))
    loaded_pth = torch.load(model_path, map_location=opt.eval_device)
    epoch = loaded_pth['epoch']

    model.load_state_dict(loaded_pth['model'], strict=strict_load_pth)
    model.eval()

    model.to(opt.eval_device)
    
    ##
    if opt.able_ret:
        memory_bank=load_clip_memory_bank(opt)
        memory_bank['vid_sent_embeds']=torch.tensor(memory_bank['vid_sent_embeds']).to('cuda')

    else:
        memory_bank=None 
    ##
    if opt.eval_mode == 'test':
        out_json_path = os.path.join(folder_path, 'dvc_results.json')
        evaluate(model,memory_bank, criterion, contrastive_criterion, postprocessors, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=False, skip_lang_eval=True,infer_mode=True)
    else:
        out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_alpha{}.json'.format(
            time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime()) + str(opt.id), epoch, len(loader.dataset),
            opt.ec_alpha))
        caption_scores, eval_loss = evaluate(model,memory_bank, criterion,contrastive_criterion, postprocessors, loader, out_json_path,
                         logger, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version, device=opt.eval_device, debug=False, skip_lang_eval=False,infer_mode=True)
        avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items() if key !='tiou'}
        avg_eval_score2 = {key: np.array(value).mean() * 4917 / len(loader.dataset) for key, value in caption_scores.items() if key != 'tiou'}

        logger.info(
            '\nValidation result based on all 4917 val videos:\n {}\n avg_score:\n{}'.format(
                                                                                       caption_scores.items(),
                                                                                       avg_eval_score))

        logger.info(
                '\nValidation result based on {} available val videos:\n avg_score:\n{}'.format(len(loader.dataset),
                                                                                           avg_eval_score2))

    logger.info('saving reults json to {}'.format(out_json_path))
    return out_json_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    
    parser.add_argument('--eval_mode', type=str, default='eval', choices=['eval', 'test'])
    parser.add_argument('--test_video_feature_folder', type=str, nargs='+', default='-')
    parser.add_argument('--test_video_meta_data_csv_path', type=str, default=None)
    parser.add_argument('--eval_folder', type=str, default='-')
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021'])
    parser.add_argument('--eval_caption_file', type=str, default='-')
    parser.add_argument('--eval_proposal_type', type=str, default='gt')
    parser.add_argument('--eval_transformer_input_type', type=str, default='queries', choices=['gt_proposals', 'queries'])
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--show_all_results', default=True)
    parser.add_argument('--eval_enable_matching_score', action='store_true', default=False)
    parser.add_argument('--eval_matching_score_weight', type=float, default=0.)
    parser.add_argument('--eval_ec_alpha', type=float, default=-1, help='-1 means using the ec_alpha from the pretrained model, while other values means using a new ec_alpha')
    parser.add_argument('--eval_calculate_query_counts', action='store_true', default=False)
    # For grounding
    parser.add_argument('--eval_enable_grounding', default=False)
    parser.add_argument('--eval_enable_maximum_matching_for_grounding', action='store_true', default=False)
    parser.add_argument('--eval_set_cost_class', type=float, default=0.)
    parser.add_argument('--eval_grounding_cost_alpha', type=float, default=0.25)
    parser.add_argument('--eval_grounding_cost_gamma', type=float, default=2)
    parser.add_argument('--eval_set_cost_cl', type=float, default=1.0)
    parser.add_argument('--eval_disable_captioning', action='store_true', default=False)
    parser.add_argument('--eval_disable_contrastive', action='store_true', default=False)
    parser.add_argument('--eval_gt_file_for_grounding', type=str, default='data/anet/captiondata/grounding/val1_for_grounding.json')
    parser.add_argument('--eval_for_multi_anno', action='store_true', default=False)
    parser.add_argument('--eval_enable_zeroshot_tal', action='store_true', default=False)
    parser.add_argument('--eval_prompt', type=str, default='a video of')
    parser.add_argument('--eval_use_amp', action='store_true', default=False)
    parser.add_argument('--eval_debug', action='store_true', default=False)
    parser.add_argument('--eval_num_queries', type=int, default=0)
    parser.add_argument('--eval_not_strict_load', action='store_true', default=False)

    parser.add_argument("--able_ret",action="store_true", default=False,help="if true, retrieval")
    parser.add_argument("--soft_k", type=int,default=50)
    parser.add_argument('--sim_match', type=str, default='cosine',help='avg_pooling, attention_avg, ... ')
    parser.add_argument('--sim_attention', type=str, default='cls_token',help='avg, cls_token, ... ')
    parser.add_argument("--proj_use",action="store_true", default=False,help="if true, downProj")
    parser.add_argument("--nvec_proj_use",action="store_true", default=False,help="if true,new vetor that made after soft attention will be downProj")
    parser.add_argument('--memory_type', type=str, default='clip',help='clip,video')
    parser.add_argument('--exp_name', type=str,help='save_path, cfgname + exp_name')
    parser.add_argument('--ret_ablation', type=str,default="None",help='ablation:Measure retrieval performance for upper bound / cheating_ablation : Measure cheated retrieval performance for upper bound ')
    parser.add_argument("--ideal_test",action="store_true", default=False,help="if true,ideal_test")
    parser.add_argument('--ret_vector', type=str, default='nvec',help='nvec,memvec')
    parser.add_argument('--ret_encdec_ref_num', type=int, default=10,help='num of reference when using memory vector retrieval')
    parser.add_argument('--down_proj', type=str, default='deep',help='simple,deep')
    parser.add_argument('--ret_text', type=str, default='token',help='token,sentence// token level or sentence level retireval')  
    parser.add_argument('--ret_loss', type=bool, default=False,help='ret loss')  
    parser.add_argument('--ret_token_encoder', type=str, default="off",help='on/off - token level embedding encoding')  
    parser.add_argument('--window_type', type=str, default='FHQ',help='FHQ,FH,FQ,HQ,Q,H,F')
    parser.add_argument('--window_avg_pooling', type=str, default='off',help='on / off')
    parser.add_argument('--text_crossAttn', action="store_true", default=True,help='True-on / False-off')
    parser.add_argument('--text_crossAttn_loc', type=str, default='before',help='before / after')
    parser.add_argument('--combined_encoder',action="store_true", default=False,help='True-concat and encode / False-seperate and encode each')
    parser.add_argument('--decoder_query_ablation', type=bool, default=False,help='True-on / False-off')
    parser.add_argument("--decoder_query_num", type=int,default=5)
    parser.add_argument("--target_domain", type=str , default='anet', help="which domain will be used in ret bank // ['anet','yc2','image']")
    parser.add_argument("--bank_type", nargs='+', default=['anet'], help="which domain will be used in ret bank // ['anet','yc2','image']")
    parser.add_argument('--window_size', type=int, default=10,help='window number')
    parser.add_argument("--retrieval_ablation", type=str , default='None', help="ideal,no_ret,none")
    parser.add_argument("--ideal_type", type=str , default='None', help="bypass_gt,bypass_pred,none")
    parser.add_argument("--ablation", type=int , default=0, help="bypass_gt,bypass_pred,none")
    parser.add_argument("--ablation_memory_size", type=int , default=100, help="n%")
    parser.add_argument("--seed_ablation", type=int , default=0, help=";")
    
    
    opt = parser.parse_args()
    opt.cap_num_feature_levels=5
    
    #debug
    # opt.target_domain='anet'
    opt.target_domain='yc2'
    # opt.bank_type=['anet']
    opt.bank_type=['yc2']
    opt.window_size=50
    # opt.window_size=20
    opt.soft_k=80
    # opt.soft_k=100
    opt.text_crossAttn=True
    opt.text_crossAttn_loc='after'
    opt.sim_match='window_cos'
    opt.ret_text='token'
    opt.down_proj='deep'
    opt.eval_folder= 'yc2_clip_cm2' 
    # opt.eval_folder= 'anet_clip_cm2' 
    opt.exp_name = 'best' 
    opt.eval_transformer_input_type ='queries' 
 
    opt.bank_path='data/bank'
    if opt.eval_folder=='yc2_clip_cm2':
        opt.test_video_feature_folder='data/yc2/features/clipvitl14.pth/'
        opt.eval_caption_file='data/yc2/captiondata/val.json'
        opt.eval_gt_file_for_caption=['data/yc2/captiondata/val.json']
        opt.target_domain='yc2'
    elif opt.eval_folder=='anet_clip_cm2':
        opt.test_video_feature_folder='data/anet/features/clipvitl14.pth/'
        opt.eval_caption_file='data/anet/captiondata/val_1.json'
        opt.eval_gt_file_for_caption=['data/anet/captiondata/val_1.json','data/anet/captiondata/val_2.json']
        opt.target_domain='anet'
    
   
   
    if not opt.text_crossAttn:
        opt.cap_num_feature_levels=4
    if opt.decoder_query_ablation:
        opt.num_queries=opt.decoder_query_num
    if opt.combined_encoder:
        opt.cap_num_feature_levels=5
    opt.eval_folder=opt.eval_folder+'_'+opt.exp_name
    
    
    opt.rec_encdec=True
    print("Rename model folder...",opt.eval_folder)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    
    if True:
        torch.backends.cudnn.enabled = False
    main(opt)