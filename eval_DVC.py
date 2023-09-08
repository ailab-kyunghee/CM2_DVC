
import collections
import json
import argparse
import statistics
def eval_metrics(dvc_filename, gt_filenames, dvc_eval_version='2018'):
    score = collections.defaultdict(lambda: -1)
    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames, version=dvc_eval_version)
    dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    # dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    # dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    score.update(dvc_score)
    return score




def predict_to_DVCjson(args):
    from collections import defaultdict

    with open(args.predict_path) as file:
        predict = json.load(file)
    sentences = []
    
    # Iterate over each video object in the JSON
    for video_data in predict:
        sentences.append(video_data["caption"])

    predicted_sentence=sentences
  
    with open(args.label_path, 'r') as file2:
        metadata_data = json.load(file2)

    # Create a dictionary to store all sentences grouped by video_id
    grouped_sentences = defaultdict(list)
    j=0
    # Iterate through the metadata data and extract sentences, video_id, duration, and timestamps
    for video_id, video_info in metadata_data.items():
        duration = video_info['duration']
        timestamps = video_info['timestamps']
        sentences = video_info['sentences']
        
        # Iterate through the sentences for this video
        for i in range(len(sentences)):
            start_time, end_time = timestamps[i]
            sentence = sentences[i].strip()  # Remove leading/trailing spaces


            # Create a dictionary for each sentence with associated information
            sentence_info = {
                # "video_id": video_id,
                "vid_duration": duration,
                "timestamp": [start_time, end_time],
                "raw_box": [start_time, end_time],
                "sentence": predicted_sentence[j],
                # "proposal_score": 0.,
                # "sentence_score": 0.,
                # "query_id": 0,
                # "pred_event_count": 0,
                # "joint_score": 0.,
            }

            # Append the sentence information to the list associated with the video_id
            grouped_sentences[video_id].append(sentence_info)
            j+=1
    # Convert the grouped_sentences defaultdict to a regular dictionary
    grouped_sentences = dict(grouped_sentences)
    
    grouped_sentences = {"results":grouped_sentences,"version":"VERSION 1.0","external_data": {"used:": True,"details":None}
                         }
    save_path='/data/minkuk/caption/DVCusingGT'
    with open(save_path, 'w') as output_file:
        json.dump(grouped_sentences, output_file, indent=2)
    return save_path


from densevid_eval3.evaluate2018 import main as eval2018
from densevid_eval3.evaluate2021 import main as eval2021

def eval_dvc(json_path, reference, no_lang_eval=False, topN=1000, version='2018'):
    args = type('args', (object,), {})()
    args.submission = json_path
    args.max_proposals_per_video = topN
    args.tious = [0.3,0.5,0.7,0.9]
    # args.verbose = False
    args.verbose = True
    args.no_lang_eval = no_lang_eval
    args.references = reference
    eval_func = eval2018 if version=='2018' else eval2021
    score = eval_func(args)
    
    avg_score={"Bleu_1":statistics.mean(score["Bleu_1"]),
               "Bleu_2":statistics.mean(score["Bleu_2"]),
               "Bleu_3":statistics.mean(score["Bleu_3"]),
               "Bleu_4":statistics.mean(score["Bleu_4"]),
               "METEOR":statistics.mean(score["METEOR"]),
               "ROUGE_L":statistics.mean(score["ROUGE_L"]),
               "CIDEr":statistics.mean(score["CIDEr"]),
               "Recall":statistics.mean(score["Recall"]),
               "Precision":statistics.mean(score["Precision"])
               }
    print(avg_score)
    return score


parser = argparse.ArgumentParser(description='Evaluation with captioning metirc')
parser.add_argument("--label_path", type=str, default="/local_datasets/caption/anet/val_1.json", help="label saved")
parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/08-04_16-23_val_preds.json", help="prediction saved")
parser.add_argument("--cap_task", type=str, default="dense",help="dense / parag")
parser.add_argument("--proposal_type", type=str, default="Prediction",help="GT,Prediction")
parser.add_argument("--metricGT_path", nargs='+', default=['/local_datasets/caption/anet/val_1.json','/local_datasets/caption/anet/val_2.json'], help="GT file path. ")
    

args = parser.parse_args()



if args.proposal_type == "GT":
    save_path = predict_to_DVCjson(args)

    if args.cap_task == 'dense':
        eval_metrics(save_path,args.metricGT_path)

else:
    if args.cap_task == 'dense':
        eval_metrics(args.predict_path,args.metricGT_path)