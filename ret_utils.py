import os
import numpy as np

def load_clip_memory_bank(args):
    first_iter=True
    for bank_type in args.bank_type:
        print("##########################################",bank_type)
        if bank_type == "anet" or bank_type =="yc2":  
            rag = os.path.join(args.bank_path,bank_type) #which domain will be used
            rag = os.path.join(rag,'clip')
            
            # rag = os.path.join(rag,'clip')
            text_sentence = np.load(os.path.join(rag,bank_type+"_scene_sentences.npy"))
            text_embed = np.load(os.path.join(rag,bank_type+"_clip_token_embeds.npy"))
            scene_videoID = np.load(os.path.join(rag,bank_type+"_scene_videoID.npy"))
        else:
            rag = os.path.join(args.bank_path,'knowledge') #which domain will be used
            text_sentence = np.load(os.path.join(rag,bank_type+"_scene_sentences.npy"))
            text_embed = np.load(os.path.join(rag,bank_type+"_clip_token_embeds.npy"))
            scene_videoID = np.load(os.path.join(rag,bank_type+"_scene_videoID.npy"))
        if first_iter:
            text_sentences = text_sentence
            scene_videoIDs = scene_videoID
            text_embeds = text_embed
        else:
            text_sentences = np.concatenate((text_sentences, text_sentence))
            text_embeds = np.concatenate((text_embeds, text_embed))
            scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
        first_iter=False    

    pair_bank={}
    
    pair_bank = {
            "vid_sentences": text_sentences,
            "vid_sent_embeds": text_embeds,
            "video_id": scene_videoIDs
        }
    print("memory loaded, ",len(pair_bank["vid_sent_embeds"]))
    return pair_bank

