import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from src.utils import feature_load
from src.utils import re_PARAselectCaption
from src.utils import DVCselectBertTopKCaptions
from src.utils import DVCselectTopKCaptions
from src.utils import PARAprep_strings
from src.utils import DVCprep_strings
from src.utils import load_data_for_PARAGinfer
from src.utils import load_data_for_DVCinfer
from src.utils import postprocess_preds_para

from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datetime
from sentence_transformers import SentenceTransformer, util

from src.utils import load_data_for_inference, prep_strings, postprocess_preds
ImageFile.LOAD_TRUNCATED_IMAGES = True

CURRENT = datetime.datetime.now().strftime("%m-%d_%H-%M")

PAD_TOKEN = '!'
EOS_TOKEN = '.'
PARAG_EOS_TOKEN = '_'
CAPTION_LENGTH = 50

# def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
#     """Models without retrival augmentation can be evaluated with a batch of length >1."""
#     out = []
#     bs = args.batch_size

#     for idx in tqdm(range(0, len(eval_df), bs)):
#         file_names = eval_df['file_name'][idx:idx+bs]
#         image_ids = eval_df['image_id'][idx:idx+bs]
#         decoder_input_ids = [PARAprep_strings('', tokenizer, is_test=True) for _ in range(len(image_ids))] 
                
#         # load image 
#         images = [Image.open(args.images_dir + file_name).convert("RGB") for file_name in file_names]
#         pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
#         with torch.no_grad():
#             preds = model.generate(pixel_values.to(args.device), 
#                                decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
#                                **args.generation_kwargs)
#         preds = tokenizer.batch_decode(preds)
 
#         for image_id, pred in zip(image_ids, preds):
#             pred = postprocess_preds_para(pred, tokenizer)
#             out.append({"image_id": int(image_id), "caption": pred})

#     return out


def predict_to_DVCjson(predict,predict_save):
    from collections import defaultdict

    # Your original JSON data

    # Create a dictionary to group entries by video ID
    grouped_data = defaultdict(list)

    # Group entries by video ID
    for entry in predict:
        video_id = entry["video_id"]
        sentence_info = {
            "vid_duration": entry["duration"],
            "timestamp": [entry["timestamp"][0],entry["timestamp"][1]],
            "raw_box": [entry["timestamp"][0],entry["timestamp"][1]],
            "sentence": entry["caption"],
            "num_clips": entry["num_clips"]
        }
        # print(sentence_info)
        
        grouped_data[video_id].append(sentence_info)
        # print(grouped_data[video_id])

   
    # Convert the grouped_sentences defaultdict to a regular dictionary
    grouped_sentences = dict(grouped_data)
    
    grouped_sentences = {"results":grouped_sentences,"version":"VERSION 1.0","external_data": {"used:": True,"details":None}
                         }
    # print(grouped_sentences)
    
    # print("##############")
    # print(grouped_sentences)
    with open(predict_save, 'w') as outfile:
            json.dump(grouped_sentences, outfile,indent=2)
    
    # with open('all_sentences_with_predictions.json', 'w') as output_file:
    #     json.dump(grouped_sentences, output_file, indent=2)
        
    # Now, grouped_sentences contains a dictionary where each key is a video_id, and the corresponding value
    # is a list of dictionaries containing video_id, duration, timestamp, sentence, and predicted_sentence


def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df,pair_bank,feature_dir_path,vid_post,prev=False,sentence_model=None,encoded_ret=None):
    """RAG models can only be evaluated with a batch of length 1."""
    if args.cap_task == "parag":
        template = open(args.parag_template_path).read().strip() + ' '

        out = []
        for idx in tqdm(range(len(eval_df))):
            # if idx == 20:
            #     break
            video_id = eval_df['video_id'][idx]
            text = eval_df['caption_gt'][idx]
            
            
            
            video_name=video_id
            if len(vid_post)==2:
                if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                    video_name=video_id[2:]
            
            feature_path = os.path.join(feature_dir_path,video_name)
            encoder_outputs = feature_load(feature_path,vid_post)
            
            num_frames=encoder_outputs.shape[0]       
            duration = eval_df['duration'][idx]
            timestamps = eval_df['timestamps'][idx]
            frame_stamps=[]
            scene_feature=[]
            for timestamp in timestamps:
                start_time = timestamp[0]
                end_time = timestamp[1]
                start_frame = int(start_time / (duration / num_frames))
                end_frame = int(end_time / (duration / num_frames))
                frame_stamps.append([start_frame, end_frame])
                scene_feature.append(np.mean(encoder_outputs[start_frame:end_frame+1],axis=0))
            scene_features = np.stack(scene_feature,axis=0)
            
            # 실험해보니까 문장이 하나밖에 생성이 안됐는데 여기를 고쳐야 할 수 도 있다. 
            # 여기 세팅을 mean 하지 않는 방향으로 한번 테스트  
            encoder_outputs = np.mean(scene_features,axis=0)
            
            # print(scene_features.shape)
            
            if args.decoder_name == "gpt2":
                linear_layer = nn.Linear(encoder_outputs.shape[1], 768)
            elif args.decoder_name == "gpt2-medium":
                linear_layer = nn.Linear(encoder_outputs.shape[1], 1024)
            elif args.decoder_name == "gpt2-large":
                linear_layer = nn.Linear(encoder_outputs.shape[1], 1280)
            elif args.decoder_name == "gpt2-xl":
                linear_layer = nn.Linear(encoder_outputs.shape[1], 1600)
            
            #when using mean
            encoder_outputs = linear_layer(torch.tensor(encoder_outputs))
            
            #when not using mean
            #encoder_outputs = linear_layer(torch.tensor(scene_features))
            # print(encoder_outputs.shape)
            # print("####")
            if args.disable_rag == False: 
                _,selected_caption = re_PARAselectCaption(pair_bank,scene_features,video_id,frame_stamps)
                decoder_input_ids= PARAprep_strings('', tokenizer, template=template,
                                                        selected_caption = selected_caption, k=int(args.k),is_test=True)

            else:
                decoder_input_ids= PARAprep_strings('', tokenizer, is_test=True)
            
            # load image
            if args.features_path is not None:
                encoder_last_hidden_state = torch.FloatTensor(encoder_outputs)
                
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.unsqueeze(0).to(args.device))
                with torch.no_grad():
                    pred = model.generate(encoder_outputs=encoder_outputs,
                                decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                **args.generation_kwargs)
            else:
                print("feature_path_none")
            
            
            pred = tokenizer.decode(pred[0])
            
            pred = postprocess_preds_para(pred, tokenizer)
            # print(pred)
            # exit()
            out.append({"video_id": video_id, "caption": pred})

        return out
    
    elif args.cap_task == 'dense':

        template = open(args.dvc_template_path).read().strip() + ' '
        out = []
        print_counter=0
        for idx in tqdm(range(len(eval_df))):
            # if print_counter>50:
            #     continue
            video_id = eval_df['video_id'][idx]
            seg_feature = eval_df['seg_feature'][idx]
            timestamp = eval_df['seg_stamp'][idx]
            duration = eval_df['duration'][idx]
            num_clips = eval_df['num_clips'][idx]
            if prev:
            #for getting previous caption
                if idx != 0 and (video_id==eval_df['video_id'][idx-1]):
                    prev_text=pred
                    prev_flag=1
                else:    
                    prev_text="This clip is first clip."
                    prev_flag=0
            else:
                prev_text=None
                prev_flag=None
            
            encoder_outputs = seg_feature
            # comp_encoder_outputs=np.mean(encoder_outputs,axis=0)
            if args.disable_rag == False: 
                # _,selected_caption = DVCselectCaption(pair_bank,encoder_outputs.detach().numpy(),video_id)
                if sentence_model is not None:
                    selected_caption = DVCselectBertTopKCaptions(pair_bank,encoder_outputs,video_id,args.k,prev_text,prev_flag,sentence_model,encoded_ret)
                else:
                    _,selected_caption = DVCselectTopKCaptions(pair_bank,encoder_outputs,video_id,args.k)
           
                decoder_input_ids= DVCprep_strings('', tokenizer, template=template,
                                                        selected_caption = selected_caption,previous_caption=prev_text, k=int(args.k),is_test=True)

            else:
                decoder_input_ids= DVCprep_strings('', tokenizer, is_test=True)
            
            
            # load image
            if args.features_path is not None:
                encoder_last_hidden_state = torch.FloatTensor(encoder_outputs)
                
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.unsqueeze(0).to(args.device))
                with torch.no_grad():
                    pred = model.generate(encoder_outputs=encoder_outputs,
                                decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                **args.generation_kwargs)
            else:
                print("feature_path_none")
            
            
            
            pred = tokenizer.decode(pred[0])
            
            if print_counter < 50:
                print("\n")
                print(video_id)
                print("\n")
                print(pred)
            pred = postprocess_preds(pred, tokenizer)
            
            out.append({"video_id": video_id, "caption": pred,"timestamp":timestamp,"duration":duration,"num_clips":int(num_clips)})
            print_counter+=1
        return out
    
        
def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn,pair_bank,feature_dir_path,vid_post,val_feature_dir_path):
    #using val2 datasaet in paragraph  ##refactorize is needed for compact code. 
    if args.infer_test == False and args.cap_task == 'parag':
        feature_dir_path=val_feature_dir_path
    
    prev = not args.disable_prev
    model = load_model(args, checkpoint_path)
    
    if args.bert_score:
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #Compute embedding for both lists
        encoded_ret = sentence_model.encode(pair_bank['scene_gt'].tolist(), convert_to_tensor=True)
    else:
        sentence_model=None
        encoded_ret=None
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df, pair_bank,feature_dir_path,vid_post,prev,sentence_model=sentence_model,encoded_ret=encoded_ret)
    # print(preds)
    # exit()
    predict_save = os.path.join(checkpoint_path, args.outfile_name)
    
    if args.predicted_proposal:
        predict_to_DVCjson(preds,predict_save)
    else:
        with open(predict_save, 'w') as outfile:
            json.dump(preds, outfile)
    with open((predict_save+'_info.txt'), 'w') as file:
            # Write the text content to the file
            file.write(args)
            # file.write("Not Using retrieval:\n")
            # file.write(str(args.disable_rag))
            # file.write("\n\nbank:\n")
            # file.write(args.bank_path)
            # file.write("\n\nbackbone:\n")
            # file.write(feature_dir_path)
            # file.write("\n\ncheckpoint\n")
            # file.write(checkpoint_path)
            # file.write("\n\nDecoder name:\n")
            # file.write(checkpoint_path)
            
def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

def main(args):
    print(args)
    if args.backbone == 'i3d':
        args.featdim = 2048
    elif args.backbone == 'tsn':
        args.featdim = 3072
    elif args.backbone == 'tsp':
        args.featdim = 512
    elif args.backbone == 'clip':
        args.featdim = 768
    print(args.decoder_name)
    print(args.cap_task)
    print(args.model_path)
    print(args.checkpoint_path)

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.infer_test or args.disable_rag:
    #     args.features_path = None
    
    if args.features_path is not None:
        feature_extractor = None
    else:
        feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k=0
        #infer_fn = evaluate_norag_model
        infer_fn = evaluate_rag_model
    else:
        infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'val'
        
        
    if args.cap_task=='dense':
        data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path = load_data_for_DVCinfer(args, args.bank_path)
        # args.k=2
    elif args.cap_task=='parag':
        data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path = load_data_for_PARAGinfer(args, args.bank_path)
    
    

    eval_df = pd.DataFrame(data[split])
    args.outfile_name = '{}_{}_preds.json'.format(CURRENT,split)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    
    if args.cap_task == 'dense':
        tokenizer.eos_token = EOS_TOKEN
    elif args.cap_task == 'parag':
        tokenizer.eos_token = PARAG_EOS_TOKEN
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn,pair_bank,feature_dir_path,vid_post,val_feature_dir_path)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn,pair_bank,feature_dir_path,vid_post,val_feature_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="data/images/", help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default='/local_datasets/caption/anet/tsp/test', help="image_features") ##path 자동화 필요
    parser.add_argument("--annotations_path", type=str, default="/local_datasets/caption/anet/val_2.json", help="JSON file with annotations in Karpathy splits")
        
    parser.add_argument("--model_path", type=str, default="/data/minkuk/caption/experiments/08-28_18-19_finetune:False_rag_7M_gpt2_30epochs", help="Path to model to use for inference")
    #parser.add_argument("--model_path", type=str, default="/data/minkuk/caption/experiments/07-17_05-48_norag_7M_gpt2", help="Path to model to use for inference")
    
    #gpt2-medium 40 epochs 7M 
    #parser.add_argument("--model_path", type=str, default="/data/minkuk/caption/experiments/07-30_18-52_finetune:True_rag_7M_gpt2-medium_40epochs", help="Path to model to use for inference")
    
    #gpt2-medium 40 epochs 28M 
    # parser.add_argument("--model_path", type=str, default="/data/minkuk/caption/experiments/07-30_18-53_finetune:True_rag_28.0M_gpt2-medium_40epochs", help="Path to model to use for inference")
    
    #parser.add_argument("--model_path", type=str, default="/data/minkuk/caption/experiments/07-19_17-19_finetune:False_rag_7M_gpt2_20epochs", help="Path to model to use for inference")
    #frozen
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint-12285", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")
    #parser.add_argument("--checkpoint_path", type=str, default="checkpoint-3130", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")
    
    #frozen
    #parser.add_argument("--checkpoint_path", type=str, default="checkpoint-12520", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=1, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")


    parser.add_argument("--base_path", type=str, default="/local_datasets/caption/", help="Directory where all data saved")
    parser.add_argument("--dataset", type=str, default="anet", help="anet, yc2")
    parser.add_argument("--bank_path",type = str, default = '/local_datasets/caption/bank')
    parser.add_argument("--bank_type", nargs='+', default=['anet'], help="which domain will be used in ret bank // ['anet','yc2','image']")
    
    parser.add_argument("--backbone", type=str, default="tsp", help="i3d,tsp,tsn,clip")
    parser.add_argument("--mode", type=str, default="test", help="train,val,test") #in saving bank, only train 
    parser.add_argument("--save_path",type=str,default="/data/minkuk/caption/result")
    parser.add_argument("--cap_task", type=str, default="dense",help="dense / parag")
    
    parser.add_argument("--parag_template_path", type=str, default="src/template.txt", help="TXT file with template")
    parser.add_argument("--dvc_template_path", type=str, default="src/dvc_template.txt", help="TXT file with template")
    parser.add_argument("--disable_prev", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--featdim", type=int,default=512)
    parser.add_argument("--bert_score",action="store_true",default=False,help="Whether you use bert score or not when selecting retrieval caption ")
    parser.add_argument("--predicted_proposal",type=str,default="PDVC",help="which proposal you use / GT or PDVC or TAL")
    args = parser.parse_args()

    main(args)