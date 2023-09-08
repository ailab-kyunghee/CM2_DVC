import collections
import logging
from pycocoevalcap.bleu.bleu import Bleu
import json
import argparse
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np
import re
from nltk.tokenize import sent_tokenize
# from nltk.tokenize import PTBTokenizer
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
def evaluate_dvc(predicted_captions,
                  gt_captions):
  """Segment-level evaluation.

  Args:
   predicted_caption: A list of strings (sentence).
   gt_caption: A list of lists (multi-ref) of strings (sentence).

  Returns:
    metrics: The NLP metrics of the predictions computed at the corpus level.
  """
  
  

  tokenizer = PTBTokenizer()
  # if self.verbose:
  #     self.scorers = [
  #         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
  #         (Meteor(),"METEOR"),
  #         (Rouge(), "ROUGE_L"),
  #         (Cider(), "CIDEr")
  #     ]
  with open(predicted_captions) as file:
    predict = json.load(file)
  sentences = []

  # Iterate over each video object in the JSON
  for video_data in predict:
      sentences.append(video_data["caption"])

  predicted_captions=sentences

  
  with open(gt_captions) as file:
    gt = json.load(file)

    sentences = []

    # Iterate over each video in the JSON
    for video_data in gt.values():
        sentence = video_data["sentences"]
        sentences.extend(sentence)
        
    gt_captions = sentences

    # Load sentences while keeping wrap sentences from the same video together
    all_preds = {}
    all_gts = {}
    video_id = 0

  for i, (preds, gts) in enumerate(zip(predicted_captions, gt_captions)):
      # all_preds[str(i)] = [' '.join(parse_sent(preds))]
      # all_gts[str(i)] = [' '.join(parse_sent(gt)) for gt in gts]
      all_gts[str(i)] = [' '.join(parse_sent(gts))]

  scorers = {
      'CIDER': Cider(),
      'METEOR': Meteor(),
  }
  
  
  metrics = collections.defaultdict(list)
  for scorer_name, scorer in scorers.items():
    score = scorer.compute_score(all_gts, all_preds)
    score = np.nan_to_num(score[0])
    metrics['Para_' + scorer_name] = float(score)

  logging.info('Closing Meteor')
  with scorers['METEOR'].lock:
    scorers['METEOR'].meteor_p.stdin.close()
    scorers['METEOR'].meteor_p.stdout.close()
    scorers['METEOR'].meteor_p.kill()
    scorers['METEOR'].meteor_p.wait()
  del scorers

  return metrics


# from nltk.tokenize.stanford import PTBTokenizer


def evaluate_cocodvc(predicted_captions,
                  gt_captions):
  """Segment-level evaluation.

  Args:
   predicted_caption: A list of strings (sentence).
   gt_caption: A list of lists (multi-ref) of strings (sentence).

  Returns:
    metrics: The NLP metrics of the predictions computed at the corpus level.
  """
  
  
  from nlgeval import NLGEval
  nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)
   
  #     self.scorers = [
  #         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
  #         (Meteor(),"METEOR"),
  #         (Rouge(), "ROUGE_L"),
  #         (Cider(), "CIDEr")
  #     ]
  with open(predicted_captions) as file:
    predict = json.load(file)
  sentences = []
  i=0
  # Iterate over each video object in the JSON
  for video_data in predict:
      sentences.append(video_data["caption"])
      # print(i)
      # i+=1
  print("a")
  predicted_captions=sentences

  
  with open(gt_captions) as file:
    gt = json.load(file)

  sentences = []

  # Iterate over each video in the JSON
  for video_data in gt.values():
      sentence = video_data["sentences"]
      sentences.extend(sentence)
      print(i)
      i+=1
      
  gt_captions = sentences
  print(len(predicted_captions))
  print(len(gt_captions))
  # Evaluate
  metrics_nlg = nlgEvalObj.compute_metrics(ref_list=gt_captions, hyp_list=predicted_captions)
  metrics_nlg = nlgEvalObj.compute_individual_metrics(ref=gt_captions[0], hyp=predicted_captions[0])
  Bleu_1 = metrics_nlg["Bleu_1"]
  Bleu_2 = metrics_nlg["Bleu_2"]
  Bleu_3 = metrics_nlg["Bleu_3"]
  Bleu_4 = metrics_nlg["Bleu_4"]
  METEOR = metrics_nlg["METEOR"]
  ROUGE_L = metrics_nlg["ROUGE_L"]
  CIDEr = metrics_nlg["CIDEr"]
  print(CIDEr)
  print(METEOR)


    # Load sentences while keeping wrap sentences from the same video together
  all_preds = {}
  all_gts = {}
  video_id = 0

  for i, (preds, gts) in enumerate(zip(predicted_captions, gt_captions)):
      # print(i)
      all_preds[i] = preds
      # all_gts[str(i)] = [' '.join(parse_sent(gt)) for gt in gts]
      all_gts[i] = gts
      
  # for i,video_data in predict:
  #     all_preds[i] = video_data
  #     print(video_data)
  # print(all_preds)
  # print("a")
  tokenizer = PTBTokenizer()
  print("b")
  
  res = tokenizer.tokenize(all_preds)
  gts = tokenizer.tokenize(all_gts)
  scorers = {
      'CIDER': Cider(),
      'METEOR': Meteor(),
  }
  
  
  metrics = collections.defaultdict(list)
  for scorer_name, scorer in scorers.items():
    score = scorer.compute_score(all_gts, all_preds)
    score = np.nan_to_num(score[0])
    metrics['dvc_' + scorer_name] = float(score)

  logging.info('Closing Meteor')
  with scorers['METEOR'].lock:
    scorers['METEOR'].meteor_p.stdin.close()
    scorers['METEOR'].meteor_p.stdout.close()
    scorers['METEOR'].meteor_p.kill()
    scorers['METEOR'].meteor_p.wait()
  del scorers

  return metrics




def evaluate_para(predicted_captions,
                  gt_captions):
  """Paragraph-level evaluation.

  Args:
   predicted_captions: A list of strings (paragraphs).
   gt_captions: A list of lists (multi-ref) of strings (paragraphs).

  Returns:
    metrics: The NLP metrics of the predictions computed at the corpus level.
  """
  with open(predicted_captions) as file:
    predict = json.load(file)
  sentences = []

  # Iterate over each video object in the JSON
  for video_data in predict:
      sentences.append(video_data["caption"])

  predicted_captions=sentences
  

  with open(gt_captions) as file:
    gt = json.load(file)

    sentences = []

    # Iterate over each video in the JSON
    for video_data in gt.values():
        sentence = ' '.join(video_data["sentences"])
        sentences.append(sentence)
        


    gt_captions = sentences

    # Load sentences while keeping wrap sentences from the same video together
    all_preds = {}
    all_gts = {}
    video_id = 0
    # for pred in predicted_captions:
    #     all_preds[str(video_id)] = [' '.join(sent_tokenize(pred))]
    #     video_id += 1

    # video_id = 0
    # for gt in gt_captions:
    #     all_gts[str(video_id)] = [' '.join(sent_tokenize(gt))]
    #     video_id += 1
  for i, (preds, gts) in enumerate(zip(predicted_captions, gt_captions)):
      all_preds[str(i)] = [' '.join(parse_sent(preds))]
      # all_gts[str(i)] = [' '.join(parse_sent(gt)) for gt in gts]
      all_gts[str(i)] = [' '.join(parse_sent(gts))]
  
  
  print(all_preds['40'])
  print(all_gts['40'])
  
  scorers = {
      'CIDER': Cider(),
      'METEOR': Meteor(),
  }
  
  
  metrics = collections.defaultdict(list)
  for scorer_name, scorer in scorers.items():
    score = scorer.compute_score(all_gts, all_preds)
    score = np.nan_to_num(score[0])
    metrics['Para_' + scorer_name] = float(score)

  logging.info('Closing Meteor')
  with scorers['METEOR'].lock:
    scorers['METEOR'].meteor_p.stdin.close()
    scorers['METEOR'].meteor_p.stdout.close()
    scorers['METEOR'].meteor_p.kill()
    scorers['METEOR'].meteor_p.wait()
  del scorers

  return metrics

def parse_sent(sent):
  """Sentence preprocessor."""
  res = re.sub('[^a-zA-Z]', ' ', sent)
  res = res.strip().lower().split()
  return res


  
  
  
  
__author__ = 'tylin'
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class COCOEvalCapFixed:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]
  
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        # scorers = [
        #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #     (Meteor(),"METEOR"),
        #     (Rouge(), "ROUGE_L"),
        #     (Cider(), "CIDEr"),
        #     (Spice(), "SPICE")
        # ]

        scorers = [
            (Meteor(),"METEOR"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
        
        
 

def coco_gtProposal_cap(args):
  # create coco object and coco_result object
  from pycocotools.coco import COCO
  from pycocoevalcap.eval import COCOEvalCap

  
  with open(args.predict_path) as file:
    predict = json.load(file)
  sentences = []
  
  # Iterate over each video object in the JSON
  for video_data in predict:
      sentences.append(video_data["caption"])

  predicted_captions=sentences
  

  with open(args.label_path) as file:
    gt = json.load(file)

  sentences = []

  # Iterate over each video in the JSON
  for video_data in gt.values():
      sentence = ' '.join(video_data["sentences"])
      sentences.append(sentence)
      

  gt_captions = sentences

  cap_to_json(gt_captions,predicted_captions)
  
  
  
  annotation_file = '/data/minkuk/caption/src/gt.json'
  results_file = '/data/minkuk/caption/src/predict.json'

  # create coco object and coco_result object
  coco = COCO(annotation_file)
  coco_result = coco.loadRes(results_file)

  # create coco_eval object by taking coco and coco_result
  coco_eval = COCOEvalCapFixed(coco, coco_result)

  # evaluate on a subset of images by setting
  # coco_eval.params['image_id'] = coco_result.getImgIds()
  # please remove this line when evaluating the full validation set
  coco_eval.params['image_id'] = coco_result.getImgIds()

  # evaluate results
  # SPICE will take a few minutes the first time, but speeds up due to caching
  coco_eval.evaluate()

  # print output evaluation scores
  for metric, score in coco_eval.eval.items():
      print(f'{metric}: {score*100}')
      # print(f'{metric}: {score*100:.3f}')
      
# def iter_caption_to_json(gts,preds):
#     # save gt caption to json format so thet we can call the api
#     # key_captions = [(key, json.loads(p)) for key, p in iter_caption]

#     info = {
#         'info': 'dummy',
#         'licenses': 'dummy',
#         'type': 'captions',
#     }
#     info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
#     n = 0
#     annotations = []
#     for k, cs in key_captions:
#         for c in cs:
#             annotations.append({
#                 'image_id': k,
#                 'caption': c['caption'],
#                 'id': n
#             })
#             n += 1
#     info['annotations'] = annotations
#     write_to_file(json.dumps(info), json_file)

def cap_to_DVCjson(gts,preds):

  # Replace these lists with your predicted and ground truth sentences
  predicted_captions = preds
  gt_captions = gts
  # Create a list to store image information
  images = []
  annotations = []
  results=[]
  # Loop through each predicted and ground truth pair
  for image_id, (predicted_caption, ground_truth_caption) in enumerate(zip(predicted_captions, gt_captions), 1):

      # Add the predicted caption as an annotation
      results.append({
          "image_id": image_id,
          "caption": predicted_caption
      })
  with open('/data/minkuk/caption/src/predict.json', 'w') as fp:
          json.dump(results, fp)
  
  
  
  # Loop through each predicted and ground truth pair
  for image_id, (predicted_caption, ground_truth_caption) in enumerate(zip(predicted_captions, gt_captions), 1):
      images.append({
          "license": 1,  # Replace with the actual license ID
          "file_name": f"image_filename{image_id}.jpg",  # Replace with the actual image file name
          "id": image_id
      })

      # Add the ground truth caption as an annotation
      annotations.append({
          "image_id": image_id,
          "id": len(annotations) + 1,
          "caption": ground_truth_caption
      })

  # Create the COCO format dictionary
  coco_format = {
      "info": {
          "description": "Your description of the dataset",
          "url": "URL to dataset",
          "version": "Version",
          "year": 2023,
          "contributor": "Your Name",
          "date_created": "Date of creation"
      },
      "licenses": [
          {
              "url": "URL to license",
              "id": 1,  # Replace with the actual license ID
              "name": "License Name"
          }
      ],
      "images": images,
      "annotations": annotations
  }

  # Save the dictionary as a JSON file
  with open("/data/minkuk/caption/src/gt.json", "w") as json_file:
      json.dump(coco_format, json_file)



      
def cap_to_json(gts,preds):

  # Replace these lists with your predicted and ground truth sentences
  predicted_captions = preds
  gt_captions = gts
  # Create a list to store image information
  images = []
  annotations = []
  results=[]
  # Loop through each predicted and ground truth pair
  for image_id, (predicted_caption, ground_truth_caption) in enumerate(zip(predicted_captions, gt_captions), 1):

      # Add the predicted caption as an annotation
      results.append({
          "image_id": image_id,
          "caption": predicted_caption
      })
  with open('/data/minkuk/caption/src/predict.json', 'w') as fp:
          json.dump(results, fp)
  
  
  
  # Loop through each predicted and ground truth pair
  for image_id, (predicted_caption, ground_truth_caption) in enumerate(zip(predicted_captions, gt_captions), 1):
      images.append({
          "license": 1,  # Replace with the actual license ID
          "file_name": f"image_filename{image_id}.jpg",  # Replace with the actual image file name
          "id": image_id
      })

      # Add the ground truth caption as an annotation
      annotations.append({
          "image_id": image_id,
          "id": len(annotations) + 1,
          "caption": ground_truth_caption
      })

  # Create the COCO format dictionary
  coco_format = {
      "info": {
          "description": "Your description of the dataset",
          "url": "URL to dataset",
          "version": "Version",
          "year": 2023,
          "contributor": "Your Name",
          "date_created": "Date of creation"
      },
      "licenses": [
          {
              "url": "URL to license",
              "id": 1,  # Replace with the actual license ID
              "name": "License Name"
          }
      ],
      "images": images,
      "annotations": annotations
  }

  # Save the dictionary as a JSON file
  with open("/data/minkuk/caption/src/gt.json", "w") as json_file:
      json.dump(coco_format, json_file)

       
def _predict_to_DVCjson(args):
    with open(args.predict_path) as file:
        predict = json.load(file)
    sentences = []
    
    # Iterate over each video object in the JSON
    for video_data in predict:
        sentences.append(video_data["caption"])

    predicted_sentence=sentences
  
    with open(args.label_path, 'r') as file2:
        metadata_data = json.load(file2)
    # Create a list to store all the sentences with their associated information
    all_sentences = []

    j=0
    # Iterate through the metadata data and extract sentences, video_id, duration, and timestamps
    for video_id, video_info in metadata_data.items():
        duration = video_info['duration']
        timestamps = video_info['timestamps']
        sentences = video_info['sentences']

        # Get the corresponding predictions for this video (if available)
        # if video_id in captions_data:
        #     predicted_sentences = captions_data[video_id]['predicted_sentences']
        # else:
        #     predicted_sentences = []  # Empty list if no predictions are available

        # Iterate through the sentences for this video
        for i in range(len(sentences)):
            start_time, end_time = timestamps[i]
            sentence = sentences[i].strip()  # Remove leading/trailing spaces

            # # Get the predicted sentence for this timestamp (if available)
            # if i < len(predicted_sentences):
            #     predicted_sentence = predicted_sentences[i]
            # else:
            #     predicted_sentence = ""  # Empty string if no prediction is available

            # Create a dictionary for each sentence with associated information
            sentence_info = {
                "video_id": video_id,
                "duration": duration,
                "timestamp": [start_time, end_time],
                "raw_box": [start_time, end_time],
                # "sentence": sentence,
                "predicted_sentence": predicted_sentence[j]
            }

            # Append the sentence information to the list
            all_sentences.append(sentence_info)
            j+=1

    # Now, all_sentences contains a list of dictionaries, each containing video_id, duration, timestamp, sentence,
    # and predicted_sentence
    # You can access this list as needed for further processing or save it to a JSON file, if desired

    # Print the first few sentences as an example
    for i in range(min(5, len(all_sentences))):
        print(all_sentences[i])

    # To save the list to a JSON file
    with open('all_sentences_with_predictions.json', 'w') as output_file:
        json.dump(all_sentences, output_file, indent=2)

# Print the combined data

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
                # "sentence": sentence,
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
    
    with open('all_sentences_with_predictions.json', 'w') as output_file:
        json.dump(grouped_sentences, output_file, indent=2)
    # Now, grouped_sentences contains a dictionary where each key is a video_id, and the corresponding value
    # is a list of dictionaries containing video_id, duration, timestamp, sentence, and predicted_sentence
    # grouped by video_id
    # You can access or save this dictionary as needed.

#   "version": "VERSION 1.0",
#     "external_data": {
#         "used:": true,
#         "details": null
#     },



parser = argparse.ArgumentParser(description='Evaluation with captioning metirc')
parser.add_argument("--label_path", type=str, default="/local_datasets/caption/anet/val_2.json", help="label saved")
#para , 1feature training, k feature infer, activitynet, tsp , bank tsp , 20ep  training
#parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/07-27_11-20_val_preds.json", help="prediction saved")
#para , 1feature training, 1 feature infer, activitynet, tsp , bank tsp , 20ep  training
#parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/07-27_10-07_val_preds.json", help="prediction saved")
#para,1f t 1f infer, gpt2-medium 
parser.add_argument("--predict_path", type=str, default="/data/minkuk/caption/result/08-04_16-23_val_preds.json", help="prediction saved")
parser.add_argument("--cap_task", type=str, default="dense",help="dense / parag")


args = parser.parse_args()


import ast
def json_test(args):
  gt_captions="/data/minkuk/caption/detection_result_nms0.8.json"
  
#   gts = json.loads(gt_captions)

#   with open(gt_captions, 'r') as train_dataset_files:
#     a=ast.literal_eval(train_dataset_files)
#     print(a)
  with open(gt_captions,'rb') as file:
    # a=ast.literal_eval(file)
    # print(a)
    
    
    gt = json.load(file)
    # print(gt["results"])
    sentences = []

    for video_data in gt["results"]:
        print(video_data)
    #   sentences.append(video_data["caption"])
    print(len(gt["results"]))
    
    # Iterate over each video in the JSON
    for video_data in gt.values():
        sentence = video_data["sentences"]
        sentences.extend(sentence)
        
    gt_captions = sentences
    
    
    
if args.cap_task == 'dense':
  # print(evaluate_dvc(args.predict_path,args.label_path))
  # print(coco_cap(args))
    json_test(args)
#   predict_to_DVCjson(args)
  # evalcap=COCOEvalCap(args.label_path,args.predict_path)
else:
  print(evaluate_para(args.predict_path,args.label_path))


