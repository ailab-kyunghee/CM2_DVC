
import json
import numpy as np
import os
import argparse
#from src.utils import save_avg_pooling
import torch
def feature_save(args):

    check = 1 
    # vgg_check=0
    # i3d_check=0
    # get data and proces
    dataset,save_path,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post = path_load(args)
    first_vid = True
    
    gt_cap_length=[]
    
    for video_id, video_data in dataset.items():
        
        duration = video_data["duration"]
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
  
        
        
        ############### single segment single sentence check ##### 
        # if np.array(timestamps).shape[0] != np.array(sentences).shape[0]:
        #     print(f"Timestamps : {timestamps}")
        #     print(f"Sentences : {sentences}")
        #     continue
        # else:
        #     print(check)
        #     continue
        # print("t",np.array(timestamps).shape)
        # print("s",np.array(sentences).shape)
       
        # print("############ Start ############")
        # print(f"Vid Id : {video_id}")
        # print(f"Duration : {duration}")
        # print(f"Timestamps : {timestamps}")
        # print(f"Sentences : {sentences}")
        
        # Vid Id : v__yWADgOFxP0
        # Duration : 238.38
        # Timestamps : [[0, 75.09], [75.09, 238.38]]
        # Sentences : ['A camera pans over a snowy area and leads into a man standing on a snowboard and riding down a mountain.', ' The man zooms in on himself riding down the hill and ends with him turning off the camera.']
        # Timestamp 1: 0 - 29 frames
        # Timestamp 2: 29 - 94 frames
        # Load the .npy file
        
        if args.backbone=='tsn':
            video_name=video_id[2:]
        else:
            video_name=video_id
        
        
        feature_path = os.path.join(feature_dir_path,video_name)
        if len(vid_post) == 2:
            #print(feature_path+vid_post[0])
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
                continue
        else:
            if os.path.exists(feature_path+vid_post[0]):
                feature_array_rgb = np.load(feature_path+vid_post[0])
                feature_matrix = np.matrix(feature_array_rgb)
                
            else:
                print("That video does not exist in json")
                continue
                
        
        num_frames=feature_matrix.shape[0]
        print(f"Video ID:{video_id}, Feature matirx shape:{feature_matrix.shape}")
        # print(duration/feature_matrix.shape[0])
        frame_numbers = []
        
        #set sentence_index for save sentence gt individually with segment // This index is used for indexing segment number in each video 
        sentence_index=0
        for timestamp in timestamps:
            start_time = timestamp[0]
            end_time = timestamp[1]

            start_frame = int(start_time / (duration / num_frames))
            end_frame = int(end_time / (duration / num_frames))
            frame_numbers.append([start_frame, end_frame])
            
            #average pooling for each segment. + expanded for bsize ( maybe set 1 for save sequence )
            #output=save_avg_pooling(feature_matrix[start_frame:end_frame+1].unsqueeze(dim=0))
            output=np.mean(feature_matrix[start_frame:end_frame+1],axis=0)
            #print("Scene fusion with avgpooling",output.shape)
            
            # set for stacking
            if first_vid == True:
                vid_features = output
                seg_sentences = sentences[sentence_index]
                seg_vid_names = video_id
                sentence_index+=1
                first_vid = False
                #print(vid_features.shape)
                #print(seg_sentences.shape)
                #print(len(seg_vid_names))
            else: 
                check+=1
                vid_features=np.append(vid_features,output,axis=0)
                seg_sentences = np.append(seg_sentences,sentences[sentence_index])
                seg_vid_names = np.append(seg_vid_names,video_id)
                sentence_index+=1
                # print(vid_features.shape)
                # print(seg_sentences.shape)
                # print(len(seg_vid_names))
        
        # Print the frame numbers corresponding to each timestamp
        # for i, frame_range in enumerate(frame_numbers):
        #     print(f"Timestamp {i+1}: {frame_range[0]} - {frame_range[1]} frames")
        # print("############  End  ############")
  
    print("If below three numbers are equal, 1 sentence for 1 segment")    
    print("segment number check:", vid_features.shape[0]) #i3d-37421
    print("sentence number check:", seg_sentences.shape[0]) #i3d-37421
    print("segment name number check:", len(seg_vid_names)) #i3d-37421
    #save vid feature
    np.save(os.path.join(save_path,"video_features.npy"), vid_features)
    #save gt sentences
    np.save(os.path.join(save_path,"scene_sentences.npy"),np.char.strip(seg_sentences))
    np.save(os.path.join(save_path,"scene_videoID.npy"),np.char.strip(seg_vid_names))
    
    # with open(os.path.join(save_path,"scene_sentences.txt"), "w") as file:
    #     for sentence in seg_sentences:
    #         #file.write(sentence.lstrip() + "\n")
    #         #file.write(sentence.strip() + "\n")
    #         file.write(sentence + "\n")
    # with open(os.path.join(save_path,"scene_videoID.txt"), "w") as file:
    #     for vid_name in seg_vid_names:
    #         file.write(vid_name + "\n")



def path_load(args):
    
    #set parameter
    vggish_feature_dir_path=None
    
    #set caption label path // which dataset
    cap_path=os.path.join(args.base_path,args.dataset)
    
    for filename in os.listdir(cap_path):
    # Check if the file name ends with the "json"
        if filename.endswith("json"):
            with open(os.path.join(cap_path,filename),"r") as json_file:
                dataset = json.load(json_file)
    #set pre-computed feature path // which dataset + backbone
    feature_path=os.path.join(cap_path,args.backbone,args.mode)
    if os.path.isdir(feature_path):
        
        #if backbone is i3d, we have two subdirectories. i3d and vggish 
        if args.backbone == 'i3d':
            feature_dir_path = os.path.join(feature_path, 'i3d_25fps_stack64step64_2stream_npy')
            vggish_feature_dir_path = os.path.join(feature_path, 'vggish_npy')
        elif args.backbone == 'tsn':
            feature_dir_path = os.path.join(feature_path, 'training')
        elif args.backbone == 'tsp':
            feature_dir_path = feature_path
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
        
        
         
    #set save path
    save_path = os.path.join(args.base_path,'bank')
    #if set mode is individual, save in each dataset and backbone dircetory. If not, save in knowledge
    if args.save == 'indiv':
        save_path = os.path.join(save_path,args.dataset,args.backbone)
    elif args.save == 'knowledge':
        save_path = os.path.join(save_path,args.save)
    
    return dataset,save_path,feature_path,feature_dir_path,vggish_feature_dir_path,vid_post



parser = argparse.ArgumentParser(description='Construct Feature Bank')

parser.add_argument("--base_path", type=str, default="/local_datasets/caption/", help="Directory where all data saved")
parser.add_argument("--dataset", type=str, default="anet", help="anet, yc2")
parser.add_argument("--backbone", type=str, default="i3d", help="i3d,tsp,tsn,clip")
parser.add_argument("--mode", type=str, default="train", help="train,val,test") #in saving bank, only train 
parser.add_argument("--save", type=str, default="indiv", help="indiv,knowledge")

args = parser.parse_args()

feature_save(args)

