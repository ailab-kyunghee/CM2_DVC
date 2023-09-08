from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
import os
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import random
CAPTION_LENGTH = 50
#DVC_CAPTION_LENGTH = 150
SIMPLE_PREFIX = "This Video shows "


def token_mask(prev,tokenizer):
    tokenized_sentence = tokenizer.encode(prev,add_special_tokens=True)
    num_tokens = len(tokenized_sentence)
    #num_tokens_to_mask = max(1, int(0.3 * num_tokens)) #30percent masking
    num_tokens_to_mask = max(1, int(0.3 * num_tokens))

    # Randomly select the indices of tokens to mask
    indices_to_mask = random.sample(range(1, num_tokens - 1), num_tokens_to_mask)
    # print(tokenizer.mask_token_id)
    # print(tokenizer.mask_token)
    # print("#########")
    # print(tokenizer.eos_token)
    # print(tokenizer.eos_token_id)
    # Mask the selected tokens with [MASK]
    for idx in indices_to_mask:
        tokenized_sentence[idx] = tokenizer.mask_token_id
    masked_sentence = tokenizer.decode(tokenized_sentence, skip_special_tokens=False)
    # print(tokenizer.encode(masked_sentence))
    #print(masked_sentence)
    #print(tokenized_sentence)
    # print(tokenizer.decode(50256,skip_special_tokens=False))
    return masked_sentence
def DVCprep_strings(text, tokenizer, template=None, selected_caption=None, previous_caption=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
        mask = False    
    else:
        padding = True 
        truncation = True #original
        mask = True
    
    if selected_caption is not None or previous_caption is not None:
        
        #both will be used
        if selected_caption is not None and previous_caption is not None:
            infix_ret = selected_caption[:]
            prefix = template.replace('||', infix_ret) #Similar video shows  // infix_ret(captions) // Previous video shows // This video shows // input
            infix_prev = previous_caption[:]
            if not is_test:
                infix_prev = token_mask(infix_prev,tokenizer)
            prefix = prefix.replace('++', infix_prev) #Similar video shows  // infix_ret(captions) // Previous video shows // infix_prev(prev_caption) // This video shows // input

        #only retrieval
        elif selected_caption is not None and previous_caption is None:
            
            infix_ret = selected_caption[:]
            prefix = template.replace('||', infix_ret) #Similar video shows  // infix_ret(captions) // Previous video shows // This video shows // input
            infix_prev = 'NONE' + '.'
            prefix = prefix.replace('++', infix_prev) #Similar video shows  // infix_ret(captions) // Previous video show s // NONE // This video shows // input
            
        #only temporal reference  
        elif selected_caption is None and previous_caption is not None:
            infix_ret = 'NONE' + '.'
            prefix = template.replace('||', infix_ret) #Similar video shows  // NONE // Previous video shows // This video shows // input
            infix_prev = previous_caption[:]
            infix_prev = token_mask(infix_prev,tokenizer)
            prefix = prefix.replace('++', infix_prev) #Similar video shows  // NONE // Previous video shows // infix_prev(prev_caption) // This video shows // input
            # max_length = 2*CAPTION_LENGTH

    #both will be not used (NOT retreival and NOT temporal reference )
    else:
        prefix = SIMPLE_PREFIX #This video shows // input
    
    # prefix(maybe prompt) encoding 
    #print(prefix)
    prefix_ids = tokenizer.encode(prefix)
    #len_prefix = len(prefix_ids)

    #maybe text(gt) encoding
    #text_sum = ' '.join(text) ### 꼭 체크해보기.. 이거 이렇게 조인시키는게 맞는지 
   # print(text_sum)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    #print(len(text_ids))
    
    if truncation:
        # print(len(text_ids)) #16
        if selected_caption is not None:
            text_ids = text_ids[:CAPTION_LENGTH]
            # print(len(text_ids)) #16
            #below 2line is for checking batch 
            trun_prefix = max_length-CAPTION_LENGTH
            prefix_ids = prefix_ids[:trun_prefix]
        else:
            prefix_ids = prefix_ids[:CAPTION_LENGTH]
            text_ids = text_ids[:CAPTION_LENGTH]
        #mine code## ### ##)exception handling for no_rag situation. ( text is short than CAPTION_LENGTH 25 )
        
    
    # print("#########")
    # print(len(prefix_ids))
    
    #concat prefix with    
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # if selected_caption is None and len(input_ids)<CAPTION_LENGTH:
    #         while len(text_ids) != CAPTION_LENGTH:
    #             text_ids += [tokenizer.pad_token_id]
    # maybe caption label 
    #authors # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    # print("################")
    # print(len(prefix_ids))
    # print(len(text_ids))
    # print(len(input_ids))


    len_prefix = len(prefix_ids)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    #print("input_ids before padding",input_ids)
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    # print(len(input_ids))
    # print(len(label_ids))
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids




# preprocess strings? maybe
def PARAprep_strings(text, tokenizer, template=None, selected_caption=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True #original
        #truncation = False
    
    if selected_caption is not None:
        
        infix = '\n\n'.join(selected_caption[:]) + '.'
        prefix = template.replace('||', infix) #Similar videos show  // infix(captions) // This video shows // input
     
    else:
        prefix = SIMPLE_PREFIX
    
    # prefix(maybe prompt) encoding 
    #print(prefix)
    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    #maybe text(gt) encoding
    text_sum = ' '.join(text) ### 꼭 체크해보기.. 이거 이렇게 조인시키는게 맞는지 
   # print(text_sum)
    text_ids = tokenizer.encode(text_sum, add_special_tokens=False)
    # print("#####################")
    # print(text_ids)    
    # cut with length-limit
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    # print("@@@@@@@@@@@@@@@@@@@@@")
    # print(text_ids)    
    # print("#####################")
    
    #mine code## ### ##exception handling for no_rag situation. ( text is short than CAPTION_LENGTH 25 )
    if selected_caption is None and len(text_ids)<25:
        while len(text_ids) != CAPTION_LENGTH:
            text_ids += [tokenizer.pad_token_id]
    
    #concat prefix with text
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids
    #print("input_ids before padding and adding text_ids",prefix_ids)
    #print("input_ids before padding and adding prefix_ids",text_ids)
    
    # maybe caption label 
    #authors # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    #print("input_ids before padding",input_ids)
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
        
        
            
    ##check for no rag . 
    # print("input_ids",len(input_ids))
    # print("prefix_ids",len(prefix_ids))
    # print("text_ids",len(text_ids))
    
    #print(len(input_ids))
    # if len(input_ids) != 166:
    #     print("infix size: ",len(infix))
    #     print("input_ids",len(input_ids))
    #     print("prefix_ids",len(prefix_ids))
    #     print("text_ids",len(text_ids))
    #     print(infix)
    # else:
    #     print("infix size: ",len(infix))
    #     print("input_ids",len(input_ids))
    #     print("prefix_ids",len(prefix_ids))
    #     print("text_ids",len(text_ids))
    # print("input",len(input_ids))
    # print("label",len(label_ids))
    # print("text",len(text_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids
def prep_strings(text, tokenizer, template=None, selected_caption=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if selected_caption is not None:
        infix = '\n\n'.join(bank_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX
    
    # prefix(maybe prompt) encoding 
    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    #maybe text(pred?) encoding
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # cut with length-limit
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    #concat prefix with text
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    
    # maybe caption label 
    #authors # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids
    
def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    pred = pred + '.'
    return pred
    
def postprocess_preds_para(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]

    # Split the result into sentences
    sentences = sent_tokenize(pred)

    # Remove duplicate sentences
    sentences = list(set(sentences))

    # Remove consecutive periods from each sentence
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = re.sub(r'\.{2,}', '.', sentence)
        processed_sentences.append(processed_sentence)

    # Join the processed sentences back into a single string
    processed_result = ' '.join(processed_sentences)

    return processed_result
def process_result(result):
    # Split the result into sentences
    sentences = sent_tokenize(result)

    # Remove duplicate sentences
    sentences = list(set(sentences))

    # Remove consecutive periods from each sentence
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = re.sub(r'\.{2,}', '.', sentence)
        processed_sentences.append(processed_sentence)

    # Join the processed sentences back into a single string
    processed_result = ' '.join(processed_sentences)

    return processed_result
    return pred

def old_PARAselectCaption(pair_bank,feature,video_id,frame_stamps):
    '''
    pair_bank[i] = {"scene": video_features[i],
                    "scene_gt": text_sentences[i],
                    "video_id": scene_videoID[i]}
    '''

    selected_caption=[]
    for frame_stamp in frame_stamps:
        scene_feature = np.mean(feature[frame_stamp[0]:frame_stamp[1]+1],axis=0)
        print(scene_feature.shape) #1 512
        similarities = []
        print(pair_bank['scene'].shape) #37421 512 
        print(pair_bank['scene_gt'].shape)
        for pair in pair_bank:
            # Check whether the target feature is in same video with bank feature.
            # We should not use bank_features from same video
            if pair_bank[pair]['video_id'] == video_id:
                print("Target video and Paired video are same. Then skip.")
                print(f"pair[video_id] : {pair_bank[pair]['video_id']}, video_id : {video_id}")
                continue
            else:
                similarity = cosine_similarity(np.array(scene_feature).reshape(-1,1),np.array(pair_bank[pair]['scene']).reshape(-1,1))
                similarities.append(similarity)
        
        #find top 1 and index
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        # select caption
        selected_caption.append(pair_bank[max_index]['scene_gt'])
    return selected_caption
import torch

def re_PARAselectCaption(pair_bank,feature,video_id,frame_stamps):
    # Move target features to GPU
    target_feature = torch.tensor(feature).to('cuda')

    # Initialize similarity scores and captions
    #similarity_scores = torch.zeros(len(target_feature)).to('cuda')
    similarity_scores = []
    captions_for_features = []

    # Process each feature individually
    for i, feature in enumerate(target_feature):
        # Move feature to GPU
        #feature = feature.to('cuda')

        # Calculate similarity score
        similarity_score = cosine_similarity(feature, torch.tensor(pair_bank['scene']).to('cuda'))
        # Find the index of the most similar caption for each feature
        most_similar_index = torch.argmax(similarity_score, dim=0)

        # Get the corresponding caption for each feature
        caption_for_feature = pair_bank['scene_gt'][most_similar_index]
        video_id_for_feature = pair_bank['video_id'][most_similar_index]
        # Check if the caption already exists in captions_for_features
        while caption_for_feature in captions_for_features or video_id_for_feature == video_id:
            # Set the similarity score of the current most similar caption to -1
            similarity_score[most_similar_index] = -1
            # Find the next maximum similarity score
            most_similar_index = torch.argmax(similarity_score, dim=0)
            # Get the corresponding caption for the feature
            caption_for_feature = pair_bank['scene_gt'][most_similar_index]
            video_id_for_feature = pair_bank['video_id'][most_similar_index]
        #print("left is target video, right is selected video_id",video_id,video_id_for_feature)
        # Accumulate similarity scores and captions
        similarity_scores.append(similarity_score)
        captions_for_features.append(caption_for_feature)

    # Return the similarity scores and captions for each feature
    return similarity_scores, captions_for_features

# def DVCselectCaption(pair_bank,feature,video_id):
#     # Move target features to GPU
#     target_feature = torch.tensor(feature).to('cuda')

#     # Initialize similarity scores and captions
#     #similarity_scores = torch.zeros(len(target_feature)).to('cuda')
#     similarity_scores = []
#     captions_for_features = []

#     # Process single feature

#     # Calculate similarity score
#     similarity_score = cosine_similarity(target_feature, torch.tensor(pair_bank['scene']).to('cuda'))
#     # Find the index of the most similar caption for each feature
#     most_similar_index = torch.argmax(similarity_score, dim=0)

#     # Get the corresponding caption for each feature
#     caption_for_feature = pair_bank['scene_gt'][most_similar_index]
#     video_id_for_feature = pair_bank['video_id'][most_similar_index]
    
#     # Return the similarity scores and captions for each feature
#     return similarity_scores, caption_for_feature

def DVCselectCaption(pair_bank, feature, video_id,k):
    # Move target features to GPU
    target_feature = torch.tensor(feature).to('cuda')

   

    # Calculate similarity scores
    similarity_score = cosine_similarity(target_feature, torch.tensor(pair_bank['scene']).to('cuda'))

    # Find the index of the most similar caption for the current feature
    most_similar_index = torch.argmax(similarity_score, dim=0)

    # Get the corresponding caption and video_id for the current feature
    caption_for_feature = pair_bank['scene_gt'][most_similar_index]
    video_id_for_feature = pair_bank['video_id'][most_similar_index]
    # print("###########")
    # print(video_id)
    # print(video_id_for_feature)
    # Check if the video_id is the same as video_id_for_feature
    while video_id_for_feature == video_id:
        # Set the similarity score for the skipped result to a very low value
        similarity_score[most_similar_index] = -1
        # Find the index of the next most similar caption
        most_similar_index = torch.argmax(similarity_score, dim=0)
        caption_for_feature = pair_bank['scene_gt'][most_similar_index]
        video_id_for_feature = pair_bank['video_id'][most_similar_index]
        # print("!!!!")
        # print(video_id_for_feature)
        

    # Return the similarity scores and captions for the single feature
    return similarity_score, caption_for_feature

def DVCselectTopKCaptions(pair_bank, feature, video_id, k):
    # Move target features to GPU
    target_feature = torch.tensor(feature).to('cuda')

    # Calculate similarity scores
    similarity_score = cosine_similarity(target_feature, torch.tensor(pair_bank['scene']).to('cuda'))

    # Exclude captions from the same video_id
    same_video_mask = pair_bank['video_id'] == video_id
    if torch.any(torch.tensor(same_video_mask)):
        similarity_score[same_video_mask] = -1

    # Get the indices of the top-k captions with the highest similarity scores
    topk_indices = torch.topk(similarity_score.detach().cpu(), k, dim=0).indices
    ######################### 만약 gpu 사용률이 많이 떨어지면 이 윗라인 때문일수도 있으니까 생각. 

    # Get the corresponding captions for the top-k indices
    topk_captions = pair_bank['scene_gt'][topk_indices]
    # print("######")
    # print(video_id)
    # print(topk_captions)
    # print(pair_bank['video_id'][topk_indices])
    # Combine topk_captions into a single string with a space separator
    if k ==1:
        combined_captions = topk_captions
    else:
        combined_captions = ' '.join(topk_captions)
    # print(combined_captions)
    # Return the top-k similarity scores and the combined captions
    return similarity_score[topk_indices], combined_captions


from sentence_transformers import SentenceTransformer, util

def DVCselectBertTopKCaptions(pair_bank,feature,video_id,k,prev_text,prev_flag,sentence_model,encoded_gt):
    # sentences = ["I'm happy", "I'm full of happiness"]
    # print("###################")
    same_video_mask = pair_bank['video_id'] == video_id
    
    
    # print(video_id)
    # print(label_text)
    # print(caption_for_feature)
    
    # Move target features to GPU
    target_feature = torch.tensor(feature).to('cuda')

    # Calculate similarity scores
    visual_similarity_score = cosine_similarity(target_feature, torch.tensor(pair_bank['scene']).to('cuda'))
    # print(similarity_score.shape)
    # Exclude captions from the same video_id
    # same_video_mask = pair_bank['video_id'] == video_id
    if torch.any(torch.tensor(same_video_mask)):
        visual_similarity_score[same_video_mask] = -1

    text_similarity_score=visual_similarity_score

    if prev_flag is not None and prev_flag != 0 :
        
        #Compute embedding for both lists
        embedding_1= sentence_model.encode(prev_text, convert_to_tensor=True)
        # embedding_2 = model.encode(pair_bank['scene_gt'].tolist(), convert_to_tensor=True)

        text_similarity_score=util.pytorch_cos_sim(embedding_1, encoded_gt)[0]
        if torch.any(torch.tensor(same_video_mask)):
            text_similarity_score[same_video_mask] = -1
        # most_similar_index = torch.argmax(similarity_score, dim=0)
        # most_similar_score = torch.max(similarity_score,dim=0)
        # # print(most_similar_score[0])
        
        # # Get the corresponding caption and video_id for the current feature
        # caption_for_feature = pair_bank['scene_gt'][most_similar_index]
        # video_id_for_feature = pair_bank['video_id'][most_similar_index]
    total_similarity_score =visual_similarity_score + text_similarity_score

    # Get the indices of the top-k captions with the highest similarity scores
    topk_indices = torch.topk(total_similarity_score.detach().cpu(), k, dim=0).indices
    ######################### 만약 gpu 사용률이 많이 떨어지면 이 윗라인 때문일수도 있으니까 생각. 

    # Get the corresponding captions for the top-k indices
    topk_captions = pair_bank['scene_gt'][topk_indices]
    # print("######")
    # print(video_id)
    # print(topk_captions)
    # print(pair_bank['video_id'][topk_indices])
    # print(visual_similarity_score[topk_indices])
    # print(text_similarity_score[topk_indices])
    # print(total_similarity_score[topk_indices])
    # Combine topk_captions into a single string with a space separator
    if k ==1:
        combined_captions = topk_captions
    else:
        combined_captions = ' '.join(topk_captions)
    # print(combined_captions)
    # Return the top-k similarity scores and the combined captions
    
    return combined_captions



def PARAselectCaption(pair_bank,feature,video_id,frame_stamps):
    '''
    pair_bank[i] = {"scene": video_features[i],
                    "scene_gt": text_sentences[i],
                    "video_id": scene_videoID[i]}
    '''

    selected_caption=[]
    #print(feature.shape)
    #print(frame_stamps)
    
   #for frame_stamp in frame_stamps:
    for i in range(len(frame_stamps)):
       
        
        #scene_feature = np.mean(feature[frame_stamp[0]:frame_stamp[1]+1],axis=0)
        scene_feature = feature[i]
        #print('target: ',scene_feature.shape)
        #print('memorybank: ',pair_bank['scene'])
        similarity = cosine_similarity(np.array(scene_feature),np.array(pair_bank['scene']))
        #print(similarity.shape)
        sorted_indices = np.argsort(similarity, axis=1)[::-1]
        #print(sorted_indices.shape) 1,segment numbers
        max_index = sorted_indices[0,0]
        # sorted_similarity = sorted(similarity, reverse=True)
        # max_index = similarity.index(sorted_similarity[0])
        
        if pair_bank['video_id'][max_index] == video_id:
            #print(f"c_first_pair[video_id] : {pair_bank['video_id'][max_index]}, video_id : {video_id}")

            find_flag = False
            i=1
            while find_flag == False:
                new_max_index = sorted_indices[0,i]
                if pair_bank['video_id'][new_max_index] == video_id:
                    i+=1 
                    #print(f"c_middle_pair[video_id] : {pair_bank['video_id'][new_max_index]}, video_id : {video_id}")
                elif pair_bank['scene_gt'][new_max_index] in selected_caption:
                    #print("already exist")
                    i+=1
                else:
                    selected_caption.append(pair_bank['scene_gt'][new_max_index])
                    #print(f"c_last_pair[video_id] : {pair_bank['video_id'][new_max_index]}, video_id : {video_id}, segment_index : {new_max_index}")
                    
                    find_flag = True
        else:
            find_flag = False
            i=0
            while find_flag == False:
                new_max_index = sorted_indices[0,i]
                if pair_bank['scene_gt'][new_max_index] in selected_caption:
                    #print("not_c_pair, already exist")
                    i+=1
                else:
                    #print(f"not_c__pair[video_id] : {pair_bank['video_id'][new_max_index]}, video_id : {video_id}, segment_index : {new_max_index}")
                    selected_caption.append(pair_bank['scene_gt'][new_max_index])
                    find_flag = True
    #print(selected_caption)
    return selected_caption


def _PARAselectCaption(pair_bank,feature,video_id,frame_stamps):
    '''
    pair_bank[i] = {"scene": video_features[i],
                    "scene_gt": text_sentences[i],
                    "video_id": scene_videoID[i]}
    '''

    selected_caption=[]
    for frame_stamp in frame_stamps:
       
        
        scene_feature = np.mean(feature[frame_stamp[0]:frame_stamp[1]+1],axis=0)
        
        similarity = cosine_similarity(np.array(scene_feature),np.array(pair_bank['scene']))
        #print(similarity.shape)
        sorted_indices = np.argsort(similarity, axis=1)[::-1]
        #print(sorted_indices.shape) 1,segment numbers
        max_index = sorted_indices[0,0]
        # sorted_similarity = sorted(similarity, reverse=True)
        # max_index = similarity.index(sorted_similarity[0])
        
        if pair_bank['video_id'][max_index] == video_id:
            print(f"c_first_pair[video_id] : {pair_bank['video_id'][max_index]}, video_id : {video_id}")

            find_flag = False
            i=1
            while find_flag == False:
                new_max_index = sorted_indices[0,i]
                if pair_bank['video_id'][new_max_index] == video_id:
                    i+=1 
                    print(f"c_middle_pair[video_id] : {pair_bank['video_id'][new_max_index]}, video_id : {video_id}")
                elif pair_bank['scene_gt'][new_max_index] in selected_caption:
                    print("already exist")
                    i+=1
                else:
                    selected_caption.append(pair_bank['scene_gt'][new_max_index])
                    print(f"c_last_pair[video_id] : {pair_bank['video_id'][new_max_index]}, video_id : {video_id}, segment_index : {new_max_index}")
                    
                    find_flag = True
        else:
            if pair_bank['scene_gt'][max_index] in selected_caption:
                find_flag = False
                i=1
                while find_flag == False:
                    new_max_index = sorted_indices[0,i]
                print("already exist")
            
            print(f"not_c__pair[video_id] : {pair_bank['video_id'][max_index]}, video_id : {video_id}, segment_index : {max_index}")
            
            selected_caption.append(pair_bank['scene_gt'][max_index])
    print(selected_caption)
    return selected_caption


## Traindataset SETTING
class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        
        # below this line, it will not work ( i will not use rag )
        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        
        ### this will not be used
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     bank_caps=caps, k=self.k, max_length=self.max_target_length)
        
        ### this will be used
        else:
            #this code is about " prompt " . And I need to know what is label. 
            #decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length) --> origin 
            bank_caps = self.df['bank'][idx] # memory bank reference , i need to see what sequence is in before this line . 
            decoder_input_ids, labels = prep_strings(text, self.tokenizer,bank_caps=bank_caps, max_length=self.max_target_length)
        
        # load precomputed features
        encoder_outputs = self.features[self.df['video_id'][idx]][()] # must replace video_id with activity id or youcook2 id. I think it need to be modulize
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding

class PARACapTrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25,pair_bank=None,vid_post=None,decoder_name=None):
        self.df = df
        self.tokenizer = tokenizer
        self.features = features_path
        self.pair_bank = pair_bank
        self.vid_post = vid_post
        self.decoder_name = decoder_name
        # below this line, it will not work ( i will not use rag )
        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        #text?
        text = self.df['caption_gt'][idx] #??
        video_id = self.df['video_id'][idx]

        #if backbone is tsn, get rid of 'v_' because tsn feature name does not include 'v_'
        video_name=video_id
        if len(self.vid_post)==2:
            if self.vid_post[0] == '_resnet.npy' or self.vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        
        
        feature_path = os.path.join(self.features,video_name)
        # load precomputed features
        encoder_outputs = feature_load(feature_path,self.vid_post)

        
        
        # time stamp 
        num_frames=encoder_outputs.shape[0]       
        duration = self.df['duration'][idx]
        timestamps = self.df['timestamps'][idx]
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

        #for not using avg pooling
        encoder_outputs = np.mean(scene_features,axis=0)
        if self.decoder_name == "gpt2":
            linear_layer = nn.Linear(encoder_outputs.shape[1], 768)
        elif self.decoder_name == "gpt2-medium":
            linear_layer = nn.Linear(encoder_outputs.shape[1], 1024)
        elif self.decoder_name == "gpt2-large":
            linear_layer = nn.Linear(encoder_outputs.shape[1], 1280)
        elif self.decoder_name == "gpt2-xl":
            linear_layer = nn.Linear(encoder_outputs.shape[1], 1600)
        
        #for not using avg pooling
        encoder_outputs = linear_layer(torch.tensor(encoder_outputs))
        # encoder_outputs = linear_layer(torch.tensor(scene_features))
        num_seg = len(timestamps)
        ### this will not be used
       # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
       # print(self.max_target_length)
        if self.rag: 
            _,selected_caption = re_PARAselectCaption(self.pair_bank,scene_features,video_id,frame_stamps)
            decoder_input_ids, labels = PARAprep_strings(text, self.tokenizer, template=self.template,
                                                     selected_caption = selected_caption, k=self.k, max_length=self.max_target_length)

        ### this will be used
        else:
            # print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #decoder_input_ids, labels = PARAprep_strings(text, self.tokenizer, max_length=self.max_target_length)
            decoder_input_ids, labels = PARAprep_strings(text, self.tokenizer, max_length=25)
        
        
        #################################### encoder output average pooling 되어야 함 #################
        
        
        # print("encoder_outputs:",torch.tensor(encoder_outputs).shape) # when no avg pooling (77,512)
        # print("decoder_inputs:",torch.tensor(decoder_input_ids).shape) # when no avg pooling (77,512)
        # print("labels:",torch.tensor(labels).shape) # when no avg pooling (77,512)
        
        # encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
        #             "decoder_input_ids": torch.tensor(decoder_input_ids),
        #             "labels": torch.tensor(labels)}
        encoding = {"encoder_outputs": encoder_outputs, 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}
        return encoding
    
    
    
class DVCCapTrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, prev=False, template_path=None, k=None, max_caption_length=25,pair_bank=None,decoder_name=None,sentence_model=None,encoded_gt=None):
        self.df = df
        self.tokenizer = tokenizer
        self.features = features_path
        self.pair_bank = pair_bank
        self.decoder_name = decoder_name
        # below this line, it will not work ( i will not use rag )
        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     + max_caption_length #previous
                                     )
            assert k is not None 
            self.k = k
        self.template = open(template_path).read().strip() + ' '
        self.rag = rag
        self.prev = prev
        
        self.sentence_model = sentence_model
        self.encoded_gt = encoded_gt
        # if self.decoder_name == "gpt2":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 768)
        # elif self.decoder_name == "gpt2-medium":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1024)
        # elif self.decoder_name == "gpt2-large":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1280)
        # elif self.decoder_name == "gpt2-xl":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1600)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        #text?
        text = self.df['caption_gt'][idx] #??
        video_id = self.df['video_id'][idx]
        seg_feature = self.df['seg_feature'][idx]
        
        if self.prev:
            #for getting previous caption
            if idx != 0 and (video_id==self.df['video_id'][idx-1]):
                prev_text=self.df['caption_gt'][idx-1]
                prev_flag=1
                # sampling for random previous caption reference
                # if random.random() < 0.6:
                #     prev_text=self.df['caption_gt'][idx-1]
                # else:
                #     prev_text = "This clip is first clip."
            else:    
                prev_text="This clip is first clip."
                prev_flag=0
        else:
            prev_text=None
            prev_flag=None
                    
        encoder_outputs = seg_feature
        

        if self.rag: 
            # _,selected_caption = DVCselectCaption(self.pair_bank,encoder_outputs.detach().numpy(),video_id)
            
            if self.sentence_model is not None:
                selected_caption = DVCselectBertTopKCaptions(self.pair_bank,encoder_outputs,video_id,self.k,prev_text,prev_flag,self.sentence_model,self.encoded_gt)
            else:
                _,selected_caption = DVCselectTopKCaptions(self.pair_bank,encoder_outputs,video_id,self.k)
                
            # _,selected_caption = DVCselectCaption(self.pair_bank,encoder_outputs,video_id)
            
            decoder_input_ids, labels = DVCprep_strings(text, self.tokenizer, template=self.template,
                                                     selected_caption = selected_caption,previous_caption=prev_text, k=self.k, max_length=self.max_target_length)
        else:
            #print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            decoder_input_ids, labels = DVCprep_strings(text, self.tokenizer,previous_caption=prev_text,template=self.template, max_length=50)
        # encoder_outputs = np.float32(encoder_outputs)
        # print(encoder_outputs.dtype)
        # print(torch.tensor(encoder_outputs).dtype)
        # print(self.linear_layer.weight.dtype)
        
        # encoder_outputs = self.linear_layer(torch.tensor(encoder_outputs))
        
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}
        return encoding




# DVC 기준 

def load_data_for_DVCtraining(args, rag=False):
    
    # if args.decoder_name == "gpt2":
    #     linear_layer = nn.Linear(args.featdim, 768)
    # elif args.decoder_name == "gpt2-medium":
    #     linear_layer = nn.Linear(args.featdim, 1024)
    # elif args.decoder_name == "gpt2-large":
    #     linear_layer = nn.Linear(args.featdim, 1280)
    # elif args.decoder_name == "gpt2-xl":
    #     linear_layer = nn.Linear(args.featdim, 1600)

    
    dataset,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post,val_dataset,val_feature_dir_path = path_load(args)
    #load memory bank
    if rag is not None:
        # Load video features
        first_iter=True
        for bank_type in args.bank_type: 
            rag = os.path.join(args.bank_path,bank_type) #which domain will be used
            rag = os.path.join(rag,args.backbone) #matching for backbone
            
            video_feature = np.load(os.path.join(rag,"video_features.npy"))
            #for no bottleneck
            video_feature = np.float32(video_feature)
            # video_feature = linear_layer(torch.tensor(video_feature)).detach().numpy()

            # Load text sentences from the text file
            text_sentence = np.load(os.path.join(rag,"scene_sentences.npy"))
            scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
            
            if first_iter:
                video_features = video_feature
                text_sentences = text_sentence
                scene_videoIDs = scene_videoID
            else:
                video_features = np.concatenate((video_features, video_feature))
                text_sentences = np.concatenate((text_sentences, text_sentence))
                scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
            first_iter=False    
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoIDs
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoIDs))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'train': [], 'val': []}

    for video_id, video_data in annotations.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            
            #for no bottleneck
            scene_feature = np.float32(scene_feature)
            # scene_feature = linear_layer(torch.tensor(scene_feature))
            
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
            data['train'] += samples
            i+=1
            
            
        
    for video_id, video_data in val_dataset.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(val_feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
             #for no bottleneck
            scene_feature = np.float32(scene_feature)
            # scene_feature = linear_layer(torch.tensor(scene_feature))
            
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
        data['val'] += samples
        i+=1
        
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path
def backup_load_data_for_DVCtraining(args, rag=False):
    
    dataset,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post,val_dataset,val_feature_dir_path = path_load(args)
    #load memory bank
    if rag is not None:
        # Load video features
        #for bank_type in args.bank_type: 
        video_features = np.load(os.path.join(rag,"video_features.npy"))

        # Load text sentences from the text file
        text_sentences = np.load(os.path.join(rag,"scene_sentences.npy"))
        scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
        
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoID
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoID))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'train': [], 'val': []}

    for video_id, video_data in annotations.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
            data['train'] += samples
            i+=1
            
            
        
    for video_id, video_data in val_dataset.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(val_feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_time,end_time]
            scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences, 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
        data['val'] += samples
        i+=1
        
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path

# paragraph 기준 
def load_data_for_PARAGtraining(args, rag=None):
    
    dataset,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post,val_dataset,val_feature_dir_path = path_load(args)
    #load memory bank
    if rag is not None:
        # Load video features
        video_features = np.load(os.path.join(rag,"video_features.npy"))

        # Load text sentences from the text file
        text_sentences = np.load(os.path.join(rag,"scene_sentences.npy"))
        scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
        
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoID
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoID))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'train': [], 'val': []}

    for video_id, video_data in annotations.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        ########################################### 이 아래로 num _ frames 어찌할지랑 append하는 부분 
        
        samples = []
        samples.append({'video_id': video_id, 'caption_gt': sentences, 'duration' : duration, 'timestamps' : timestamps})
        
        data['train'] += samples
    for video_id, video_data in val_dataset.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        ########################################### 이 아래로 num _ frames 어찌할지랑 append하는 부분 
        
        samples = []
        samples.append({'video_id': video_id, 'caption_gt': sentences, 'duration' : duration, 'timestamps' : timestamps})
        
        data['val'] += samples
        
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path
        
        

def load_data_for_DVCinfer(args, rag=None):
    
    # if args.decoder_name == "gpt2":
    #     linear_layer = nn.Linear(args.featdim, 768)
    # elif args.decoder_name == "gpt2-medium":
    #     linear_layer = nn.Linear(args.featdim, 1024)
    # elif args.decoder_name == "gpt2-large":
    #     linear_layer = nn.Linear(args.featdim, 1280)
    # elif args.decoder_name == "gpt2-xl":
    #     linear_layer = nn.Linear(args.featdim, 1600)
    
    
    dataset,_,feature_dir_path,_,vid_post,_,val_feature_dir_path = path_load(args)
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ rag:",rag)
    #load memory bank
    if rag is not None:
        # Load video features
        first_iter=True
        for bank_type in args.bank_type: 
            rag = os.path.join(args.bank_path,bank_type) #which domain will be used
            rag = os.path.join(rag,args.backbone) #matching for backbone
            
            video_feature = np.load(os.path.join(rag,"video_features.npy"))
            video_feature = np.float32(video_feature)
            # video_feature = linear_layer(torch.tensor(video_feature)).detach().numpy()

            # Load text sentences from the text file
            text_sentence = np.load(os.path.join(rag,"scene_sentences.npy"))
            scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
            
            if first_iter:
                video_features = video_feature
                text_sentences = text_sentence
                scene_videoIDs = scene_videoID
            else:
                video_features = np.concatenate((video_features, video_feature))
                text_sentences = np.concatenate((text_sentences, text_sentence))
                scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
            first_iter=False    
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoIDs
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoIDs))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'test': [], 'val': []}

    #for checking the short result
    c=0
    
    #using val2 datasaet
    if args.infer_test == False:
        feature_dir_path=val_feature_dir_path
    
    predicted_proposal=None
    if args.predicted_proposal == "TAL":
        tal="/data/minkuk/caption/detection_result_nms0.8.json"
        with open(tal,'rb') as file:
            tal_proposals = json.load(file)
        predicted_proposal = tal_proposals["results"]
    if args.predicted_proposal == "PDVC":
        pdvc="/data/minkuk/caption/num4917_epoch29.json_rerank_alpha1.0_temp2.0.json"
        with open(pdvc,'rb') as file:
            pdvc_proposals = json.load(file)
        predicted_proposal = pdvc_proposals["results"]
        # for video_data in gt["results"]:
        #     print(video_data)

    check_non_vid=0
    
    for video_id, video_data in annotations.items():
        
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        check_num_clips=0
        
        
        
        
        #pdvc proposals
        if video_id in predicted_proposal and args.predicted_proposal=="PDVC":

            segment_list = []

            for entry in predicted_proposal[video_id]:
                # print(entry)
                # Check if we have added enough segments
                # if len(segment_list) >= gt_num:
                #     break
                # if entry["score"] <= 0.5:
                #     break
                # check_num_clips+=1
                # segment = predicted_proposal[entry]
                segment = entry["timestamp"]
                # print(segment)
                segment_list.append(segment)
                
                
            # print(segment_list)
            timestamps=segment_list
            # print(timestamps)
        elif video_id not in predicted_proposal:
            check_non_vid +=1
            print(check_non_vid)
        
        
        
        
        
        
        if video_id[2:] in predicted_proposal and args.predicted_proposal=="TAL":
            gt_num = len(timestamps)
            # gt_num = 10
            
            segment_list = []

            for entry in predicted_proposal[video_id[2:]]:
                # print(entry)
                # Check if we have added enough segments
                # if len(segment_list) >= gt_num:
                #     break
                if entry["score"] <= 0.5:
                    break
                check_num_clips+=1
                # segment = predicted_proposal[entry]
                segment = entry["segment"]
                # print(segment)
                segment_list.append(segment)
                
                
            # print(segment_list)
            timestamps=segment_list
        elif video_id not in predicted_proposal:
            check_non_vid+=1
            print(check_non_vid)

        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_time,end_time]
            scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            #for no bottleneck
            scene_feature = np.float32(scene_feature)
            # scene_feature = linear_layer(torch.tensor(scene_feature))
            samples = []
            # samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
            samples.append({'video_id': video_id, 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature,"num_clips":check_num_clips})
            if args.infer_test:
                data['test'] += samples
            else:
                data['val'] += samples
            i+=1
        # c+=1
        # if c==5:
        #     break
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path


def load_data_for_PARAGinfer(args, rag=None):
    
    dataset,_,feature_dir_path,_,vid_post,_,val_feature_dir_path = path_load(args)
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ rag:",rag)
    #load memory bank
    if rag is not None:
        # Load video features
        video_features = np.load(os.path.join(rag,"video_features.npy"))
        # Load text sentences from the text file
        text_sentences = np.load(os.path.join(rag,"scene_sentences.npy"))
        scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
        
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoID
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoID))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'test': [], 'val': []}

    #for checking the short result
    # i=0
    
    for video_id, video_data in annotations.items():
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        samples = []
        samples.append({'video_id': video_id, 'caption_gt': sentences, 'duration' : duration, 'timestamps' : timestamps})
        if args.infer_test:
            data['test'] += samples
        else:
            data['val'] += samples
        # i+=1
        # if i == 10:
        #     break

        
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path

def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)

    return data      


def load_data_for_saving(annot_path):
    annotations = json.load(open(annot_path))['images']
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 



#avg pooling for saving bank
# def save_avg_pooling(input_tensor):
#     # Apply adaptive average pooling to the input tensor
#     # Apply global average pooling along the second dimension (axis 1)
#     pooling_result = nn.AdaptiveAvgPool1d(1)(input_tensor)
#     print(pooling_result.shape)
#     return pooling_result.squeeze(dim=1)

# 일단 그냥 train 전용  
def path_load(args):
    
    #set parameter
    vggish_feature_dir_path=None
    
    #set caption label path // which dataset
    cap_path=os.path.join(args.base_path,args.dataset)
    val_cap_path =os.path.join(args.base_path,args.dataset)
    
    if args.mode == "train":
        for filename in os.listdir(cap_path):
        # Check if the file name ends with the "json"
            if filename.startswith("train"):
                with open(os.path.join(cap_path,filename),"r") as json_file:
                    dataset = json.load(json_file)
        
        # Check if the file name ends with the "json"
        for filename in os.listdir(val_cap_path):
            if filename.startswith("val_2"):
                with open(os.path.join(cap_path,filename),"r") as json_file:
                    val_dataset = json.load(json_file)  
    elif args.mode == "test":
        val_dataset=None
        val_feature_dir_path=None
        for filename in os.listdir(val_cap_path):
            if filename.startswith("val_1"):
                with open(os.path.join(cap_path,filename),"r") as json_file:
                    dataset = json.load(json_file)  
             
    #set pre-computed feature path // which dataset + backbone
    
    feature_path=os.path.join(cap_path,args.backbone,args.mode)
    val_feature_path = os.path.join(cap_path,args.backbone,'val')
    if os.path.isdir(feature_path):
        
        #if backbone is i3d, we have two subdirectories. i3d and vggish 
        if args.backbone == 'i3d':
            feature_dir_path = os.path.join(feature_path, 'i3d_25fps_stack64step64_2stream_npy')
            vggish_feature_dir_path = os.path.join(feature_path, 'vggish_npy')
            val_feature_dir_path=os.path.join(cap_path,args.backbone,'train','i3d_25fps_stack64step64_2stream_npy') #i3d don't has seperated folders. 
        elif args.backbone == 'tsn':
            feature_dir_path = os.path.join(feature_path, 'training')
            val_feature_dir_path = os.path.join(val_feature_path, 'validation')
        elif args.backbone == 'tsp':
            feature_dir_path = feature_path
            val_feature_dir_path = val_feature_path
        elif args.backbone == 'clip':
            feature_dir_path = os.path.join(feature_path, 'none')
    
    vid_post=[]
    #post fix vid name
    if args.backbone == 'i3d':
        vid_post.append('_rgb.npy')
        vid_post.append('_flow.npy')
    elif args.backbone == 'tsn':
        vid_post.append('_resnet.npy')
        vid_post.append('_bn.npy')
    elif args.backbone == 'tsp': ## need to add like upper ones
        vid_post.append('.npy')
    elif args.backbone == 'clip':
        pass
        
        
    return dataset,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post,val_dataset,val_feature_dir_path

def feature_load(feature_path,vid_post):
    if len(vid_post) == 2:
        # print(feature_path+vid_post[0])
        # print(feature_path+vid_post[1])
        if os.path.exists(feature_path+vid_post[0]) & os.path.exists(feature_path+vid_post[1]):
            feature_array_rgb = np.load(feature_path+vid_post[0])
            feature_array_motion = np.load(feature_path+vid_post[1])
            feature_matrix_rgb = np.matrix(feature_array_rgb)
            feature_matrix_motion = np.matrix(feature_array_motion)
            feature_matrix = np.concatenate([feature_matrix_rgb,feature_matrix_motion],axis=-1)
            #print(feature_matrix.shape)
            #print("@@@@@@@@@@@@@@@@@@@")
            
        else:
            print("That video does not exist in json")
            feature_matrix=None
                
    else:
        #print(feature_path+vid_post[0])
        if os.path.exists(feature_path+vid_post[0]):
            feature_array_rgb = np.load(feature_path+vid_post[0])
            feature_matrix = np.matrix(feature_array_rgb)
            
        else:
            print("That video does not exist in json")
            feature_matrix=None

    return feature_matrix













## for no avg test in inference step

def load_data_for_DVCinfer_test(args, rag=None):
    
    if args.decoder_name == "gpt2":
        linear_layer = nn.Linear(args.featdim, 768)
    elif args.decoder_name == "gpt2-medium":
        linear_layer = nn.Linear(args.featdim, 1024)
    elif args.decoder_name == "gpt2-large":
        linear_layer = nn.Linear(args.featdim, 1280)
    elif args.decoder_name == "gpt2-xl":
        linear_layer = nn.Linear(args.featdim, 1600)
    
    
    dataset,_,feature_dir_path,_,vid_post,_,val_feature_dir_path = path_load(args)
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ rag:",rag)
    #load memory bank
    if rag is not None:
        # Load video features
        first_iter=True
        for bank_type in args.bank_type: 
            rag = os.path.join(args.bank_path,bank_type) #which domain will be used
            rag = os.path.join(rag,args.backbone) #matching for backbone
            
            video_feature = np.load(os.path.join(rag,"video_features.npy"))
            video_feature = np.float32(video_feature)
            video_feature = linear_layer(torch.tensor(video_feature)).detach().numpy()

            # Load text sentences from the text file
            text_sentence = np.load(os.path.join(rag,"scene_sentences.npy"))
            scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
            
            if first_iter:
                video_features = video_feature
                text_sentences = text_sentence
                scene_videoIDs = scene_videoID
            else:
                video_features = np.concatenate((video_features, video_feature))
                text_sentences = np.concatenate((text_sentences, text_sentence))
                scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
            first_iter=False    
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoIDs
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoIDs))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'test': [], 'val': []}

    #for checking the short result
    c=0
    
    #using val2 datasaet
    if args.infer_test == False:
        feature_dir_path=val_feature_dir_path
    
    
    for video_id, video_data in annotations.items():
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            scene_feature=encoder_outputs[start_frame:end_frame+1]
            # scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            #for no bottleneck
            scene_feature = np.float32(scene_feature)
            scene_feature = linear_layer(torch.tensor(scene_feature))
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
            if args.infer_test:
                data['test'] += samples
            else:
                data['val'] += samples
            i+=1
        # c+=1
        # if c==5:
        #     break
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path



def noavg_load_data_for_DVCtraining(args, rag=False):
    
    if args.decoder_name == "gpt2":
        linear_layer = nn.Linear(args.featdim, 768)
    elif args.decoder_name == "gpt2-medium":
        linear_layer = nn.Linear(args.featdim, 1024)
    elif args.decoder_name == "gpt2-large":
        linear_layer = nn.Linear(args.featdim, 1280)
    elif args.decoder_name == "gpt2-xl":
        linear_layer = nn.Linear(args.featdim, 1600)

    
    dataset,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post,val_dataset,val_feature_dir_path = path_load(args)
    #load memory bank
    if rag is not None:
        # Load video features
        first_iter=True
        for bank_type in args.bank_type: 
            rag = os.path.join(args.bank_path,bank_type) #which domain will be used
            rag = os.path.join(rag,args.backbone) #matching for backbone
            
            video_feature = np.load(os.path.join(rag,"video_features.npy"))
            #for no bottleneck
            video_feature = np.float32(video_feature)
            video_feature = linear_layer(torch.tensor(video_feature)).detach().numpy()

            # Load text sentences from the text file
            text_sentence = np.load(os.path.join(rag,"scene_sentences.npy"))
            scene_videoID = np.load(os.path.join(rag,"scene_videoID.npy"))
            
            if first_iter:
                video_features = video_feature
                text_sentences = text_sentence
                scene_videoIDs = scene_videoID
            else:
                video_features = np.concatenate((video_features, video_feature))
                text_sentences = np.concatenate((text_sentences, text_sentence))
                scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
            first_iter=False    
        # pairing two memory informations from bank. 
        pair_bank={}
        if len(video_features) == len(text_sentences):
            pair_bank = {
                    "scene": video_features,
                    "scene_gt": text_sentences,
                    "video_id": scene_videoIDs
                }
        else:
            print("number of pairs is not matched")
            print("scene", len(video_features))
            print("scene_gt", len(text_sentences))
            print("video_id",len(scene_videoIDs))
            return -1
    else:
        pair_bank=None

    annotations=dataset
    data = {'train': [], 'val': []}

    for video_id, video_data in annotations.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]

        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            #scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            scene_feature=encoder_outputs[start_frame:end_frame+1]
            #for no bottleneck
            scene_feature = np.float32(scene_feature)
            scene_feature = linear_layer(torch.tensor(scene_feature))
            
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
            data['train'] += samples
            i+=1
            
            
        
    for video_id, video_data in val_dataset.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        video_name=video_id
        if len(vid_post)==2:
            if vid_post[0] == '_resnet.npy' or vid_post[1] == '_resnet.npy':
                video_name=video_id[2:]
        feature_path = os.path.join(val_feature_dir_path,video_name)
        encoder_outputs = feature_load(feature_path,vid_post)
        if encoder_outputs is None:
            continue
        num_frames=encoder_outputs.shape[0]       
        
        i=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            seg_stamp=[start_frame,end_frame]
            #scene_feature=np.mean(encoder_outputs[start_frame:end_frame+1],axis=0)
            scene_feature=encoder_outputs[start_frame:end_frame+1]
             #for no bottleneck
            scene_feature = np.float32(scene_feature)
            scene_feature = linear_layer(torch.tensor(scene_feature))
            
            samples = []
            samples.append({'video_id': video_id, 'caption_gt': sentences[i], 'duration' : duration, 'timestamps' : timestamps,'n_th':i,'seg_stamp':seg_stamp,'seg_feature':scene_feature})
        
        data['val'] += samples
        i+=1
        
    return data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path

class noavg_DVCCapTrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, prev=False, template_path=None, k=None, max_caption_length=25,pair_bank=None,decoder_name=None):
        self.df = df
        self.tokenizer = tokenizer
        self.features = features_path
        self.pair_bank = pair_bank
        self.decoder_name = decoder_name
        # below this line, it will not work ( i will not use rag )
        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.template = open(template_path).read().strip() + ' '
        self.rag = rag
        self.prev = prev
        # if self.decoder_name == "gpt2":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 768)
        # elif self.decoder_name == "gpt2-medium":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1024)
        # elif self.decoder_name == "gpt2-large":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1280)
        # elif self.decoder_name == "gpt2-xl":
        #     self.linear_layer = nn.Linear(df['seg_feature'][0].shape[1], 1600)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        #text?
        text = self.df['caption_gt'][idx] #??
        video_id = self.df['video_id'][idx]
        seg_feature = self.df['seg_feature'][idx]
        
        if self.prev:
            #for getting previous caption
            if idx != 0 and (video_id==self.df['video_id'][idx-1]):
                
                # sampling for random previous caption reference
                if random.random() < 0.6:
                    prev_text=self.df['caption_gt'][idx-1]
                else:
                    prev_text = "This clip is first clip."
            else:    
                prev_text="This clip is first clip."
        else:
            prev_text=None
                    
        encoder_outputs = seg_feature
        
        comp_encoder_outputs=np.mean(encoder_outputs.detach().numpy(),axis=0)
            
            
        if self.rag: 
            # _,selected_caption = DVCselectCaption(self.pair_bank,comp_encoder_outputs,video_id,self.k)
            _,selected_caption = DVCselectTopKCaptions(self.pair_bank,comp_encoder_outputs,video_id,self.k)
            
            decoder_input_ids, labels = DVCprep_strings(text, self.tokenizer, template=self.template,
                                                     selected_caption = selected_caption,previous_caption=prev_text, k=self.k, max_length=self.max_target_length)
        else:
            #print("rag not activated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            decoder_input_ids, labels = DVCprep_strings(text, self.tokenizer,previous_caption=prev_text,template=self.template, max_length=50)
        # encoder_outputs = np.float32(encoder_outputs)
        # print(encoder_outputs.dtype)
        # print(torch.tensor(encoder_outputs).dtype)
        # print(self.linear_layer.weight.dtype)
        
        # encoder_outputs = self.linear_layer(torch.tensor(encoder_outputs))
        
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}
        return encoding

