from eval_dvc import eval_dvc
import collections
import logging
from pycocoevalcap.bleu.bleu import Bleu
import json
import argparse
def eval_pdvc(args):

  scores = eval_metrics(args.predict_path,
                            gt_filenames=args.label_path
                            )
  
  
def eval_metrics(dvc_filename, gt_filenames, para_gt_filenames=None, alpha=0.3, ranking_key='proposal_score', rerank=False, dvc_eval_version='2018'):
    score = collections.defaultdict(lambda: -1)

    # top_n = 3
    # top_n_filename = dvc_filename + '.top{}.json'.format(top_n)
    # get_topn_from_dvcjson(dvc_filename, top_n_filename, top_n=top_n, ranking_key=ranking_key)
    # dvc_score = eval_dvc(json_path=top_n_filename, reference=gt_filenames)
    # dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    # dvc_score.update(eval_soda(top_n_filename, ref_list=gt_filenames))
    # dvc_score.update(eval_para(top_n_filename, referneces=para_gt_filenames))
    # for key in dvc_score.keys():
    #     score[key] = dvc_score[key]

    
    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames, version=dvc_eval_version)
    dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    # dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    # dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    # score.update(dvc_score)
    print(dvc_score)
    return dvc_score
    # return score



parser = argparse.ArgumentParser(description='Evaluation with captioning metirc')
parser.add_argument("--label_path", type=str, default="/local_datasets/caption/anet/val_1.json", help="label saved")
#para , 1feature training, k feature infer, activitynet, tsp , bank tsp , 20ep  training
#parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/07-27_11-20_val_preds.json", help="prediction saved")
#para , 1feature training, 1 feature infer, activitynet, tsp , bank tsp , 20ep  training
#parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/07-27_10-07_val_preds.json", help="prediction saved")
#para,1f t 1f infer, gpt2-medium 
parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/08-04_16-23_val_preds.json", help="prediction saved")
parser.add_argument("--cap_task", type=str, default="dense",help="dense / parag")


args = parser.parse_args()

eval_pdvc(args)