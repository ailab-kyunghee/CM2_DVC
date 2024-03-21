
import json
import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import clip


def feature_save(args):
    check = 1 
    dataset,save_path,vid_post = path_load(args)
    first_vid = True
    clip_model, feature_extractor = clip.load("ViT-L/14", device="cuda")
    
    gt_cap_length=[]
    rescaled_features=[]
    seg_sentences=[]
    for video_id, video_data in tqdm(dataset.items()):
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
           
        if args.backbone=='tsn':
            video_name=video_id[2:]
        else:
            video_name=video_id
        
        #set sentence_index for save sentence gt individually with segment // This index is used for indexing segment number in each video 
        sentence_index=0
        for timestamp in timestamps:
            # set for stacking
            if first_vid == True:
                if args.only_token:
                    seg_sentences.append(sentences[sentence_index])
                    seg_vid_names = video_id
                    token = clip.tokenize(sentences[sentence_index],truncate=True)
                    token_embeds=np.expand_dims(clip_model.encode_text(token.to('cuda')).cpu().detach(),axis=0)
                
                
                sentence_index+=1
                first_vid = False
                
            else: 
                check+=1
                if args.only_token:
                    token = clip.tokenize(sentences[sentence_index],truncate=True)
                    adding_token=np.expand_dims(clip_model.encode_text(token.to('cuda')).cpu().detach(),axis=0)
                    token_embeds=np.append(token_embeds,adding_token,axis=0)
                    seg_sentences.append(sentences[sentence_index])
                    seg_vid_names = np.append(seg_vid_names,video_id)
                

                sentence_index+=1
  
  
    if args.dataset=="yc2":
        save_path="bank/yc2/clip"
    elif args.dataset=="anet":
        save_path = "bank/anet/clip"
    if args.only_token:
        print("sentence embedding number check:", token_embeds.shape[0]) #i3d-37421
        print("sentence embedding number check:", token_embeds.shape) #i3d-37421
        np.save(os.path.join(save_path,args.dataset+"_scene_sentences.npy"),np.char.strip(np.array(seg_sentences)))
        seg_sents={'text':seg_sentences}
        with open(os.path.join(save_path,args.dataset+"_scene_sentences.json"), 'w') as outfile:
            json.dump(seg_sents, outfile)
        np.save(os.path.join(save_path,args.dataset+"_scene_videoID.npy"),np.char.strip(seg_vid_names))
        np.save(os.path.join(save_path,args.dataset+"_clip_token_embeds.npy"),token_embeds)




def path_load(args):
    
    #set parameter
    vggish_feature_dir_path=None
    
    #set caption label path // which dataset
    cap_path=os.path.join(args.base_path,args.dataset)

    # open train json.
    if args.dataset=="anet":
        with open(os.path.join(cap_path,"captiondata/train_modified.json"),"r") as json_file:
            dataset = json.load(json_file)
        
    elif args.dataset=="yc2":
        with open(os.path.join(cap_path,"captiondata/train.json"),"r") as json_file:
            dataset = json.load(json_file)

    
    print(json_file)    
    vid_post=[]
         
    #set save path
    # hard coding. this path should be fixed
    
    save_path = 'bank'
    #if set mode is individual, save in each dataset and backbone dircetory. If not, save in knowledge
    if args.save == 'indiv':
        save_path = os.path.join(save_path,args.dataset,args.backbone)
    elif args.save == 'knowledge':
        save_path = os.path.join(save_path,args.save)
    
    return dataset,save_path,vid_post


parser = argparse.ArgumentParser(description='Construct Feature Bank')

# parser.add_argument("--base_path", type=str, default="/local_datasets/caption/", help="Directory where all data saved")
parser.add_argument("--base_path", type=str, default="data", help="Directory where all data saved")
parser.add_argument("--dataset", type=str, default="anet", help="anet, yc2")
parser.add_argument("--backbone", type=str, default="clip", help="i3d,tsp,tsn,clip")
parser.add_argument("--mode", type=str, default="train", help="train,val,test") #in saving bank, only train 
parser.add_argument("--save", type=str, default="indiv", help="indiv,knowledge")
parser.add_argument("--sent_embed",action="store_true",default=False, help="_")
parser.add_argument("--only_token",type=bool,default=True, help="_")
parser.add_argument("--sent_encoder", type=str, default=None, help="roberta,t5 ...")
parser.add_argument("--external",action="store_true",default=False, help="_")


args = parser.parse_args()

feature_save(args)
