import pandas as pd
import numpy as np
import os
import argparse
import datetime
os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from transformers import VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel,EncoderDecoderModel
from src.vision_encoder_decoder import SmallCap, SmallCapConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.opt import ThisOPTConfig, ThisOPTForCausalLM

from src.utils import *

from sentence_transformers import SentenceTransformer, util



# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
MASK_TOKEN = '-'
CAPTION_LENGTH = 200
DVC_CAPTION_LENGTH = 50
CURRENT = datetime.datetime.now().strftime("%m-%d_%H-%M")

def get_model_and_auxiliaries(args):

    # register model types
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    if args.extractor=='CLIP':
        feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    #tokenizer.add_tokens([mask_token])
    tokenizer.mask_token = MASK_TOKEN
    model = SmallCap.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name, cross_attention_reduce_factor=cross_attention_reduce_factor,featdim=args.featdim)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 

    model.config.mask_token_id = tokenizer.mask_token_id
    
    # AutoConfig.register(args.backbone, SmallCapConfig)
    
    model.config.backbone=args.backbone
    model.config.backbone_featdim = args.featdim
    
    
    if not args.disable_rag:
        if args.cap_task=='dense':
            model.config.k = args.k
        else:
            model.config.k = 37
        #model.config.retrieval_encoder = args.retrieval_encoder   
    if args.cap_task=='dense':
        model.config.max_length = DVC_CAPTION_LENGTH
    else:
        model.config.max_length = CAPTION_LENGTH   
    model.config.rag = not args.disable_rag
  
    #print("model",model)
    #print(stop)
    # freeze parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
                for name, param in model.decoder.named_parameters():
                    if 'encoder_attn' not in name:
                        param.requires_grad = False

    else:
        ### finetune or not
        if args.finetune:
            for name, param in model.decoder.named_parameters():
                param.requires_grad = True
        else: #freeze decoder too. 
            # if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    #print("set requires_grad false except crossattention")
                    param.requires_grad = False
                        

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))


    if args.extractor=='CLIP':
        return model, tokenizer, feature_extractor
    else:
        return model, tokenizer
    
def get_data(tokenizer, max_length, args):
    
    # anntotation set  //  I need not only caption, but also time stamp. And also i need to consider the change about image to video 
    # maybe i need to replace this part with vid caption model. 
    #data = load_data_for_DVCtraining(args,args.annotations_path, args.captions_path)
    if args.cap_task=='dense':
        data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path = load_data_for_DVCtraining(args, args.bank_path)
        # data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path = noavg_load_data_for_DVCtraining(args, args.bank_path)
        template_path=args.dvc_template_path
        
    elif args.cap_task=='parag':
        data,pair_bank,feature_dir_path,vid_post,val_feature_dir_path = load_data_for_PARAGtraining(args, args.bank_path)
        template_path=args.parag_template_path
        
    #there are annotation and memory caption in data. So, it is same in train_df
    train_df = pd.DataFrame(data['train'])# Final Data construct/ I need consider the args.k and max_caption_length. And what is template_path. And maybe i need to remove rag
    
    
    
    if args.cap_task=='dense':
        train_dataset = DVCCapTrainDataset(
        # train_dataset = noavg_DVCCapTrainDataset(
                            df=train_df,
                            features_path=feature_dir_path,
                            tokenizer=tokenizer,
                            rag=not args.disable_rag,
                            prev=not args.disable_prev,
                            template_path=template_path,
                            k=args.k,
                            max_caption_length=max_length,
                            pair_bank=pair_bank,
                            decoder_name=args.decoder_name
                            )
        
   
        if args.bert_score:
            sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            #Compute embedding for both lists
            encoded_ret = sentence_model.encode(pair_bank['scene_gt'].tolist(), convert_to_tensor=True)    
            train_dataset = DVCCapTrainDataset(
                            df=train_df,
                            features_path=feature_dir_path,
                            tokenizer=tokenizer,
                            rag=not args.disable_rag,
                            prev=not args.disable_prev,
                            template_path=template_path,
                            k=args.k,
                            max_caption_length=max_length,
                            pair_bank=pair_bank,
                            decoder_name=args.decoder_name,
                            sentence_model=sentence_model,
                            encoded_gt = encoded_ret
                            )
        
    elif args.cap_task == 'parag':
        train_dataset = PARACapTrainDataset(
                            df=train_df,
                            features_path=feature_dir_path,
                            tokenizer=tokenizer,
                            rag=not args.disable_rag,
                            template_path=template_path,
                            k=args.k,
                            max_caption_length=max_length,
                            pair_bank=pair_bank,
                            vid_post=vid_post,
                            decoder_name=args.decoder_name
                            )

    return train_dataset

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
    # 모델 선언부 
    print(args.decoder_name)
    if args.extractor=='CLIP':
        model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    else:
        model, tokenizer = get_model_and_auxiliaries(args)
   
    #Train Data set
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    model_type = 'norag' if args.disable_rag else 'rag'
    if args.ablation_visual:
        output_dir = '{}_{}_{}M_{}_ablation'.format(CURRENT,model_type, args.attention_size, args.decoder_name)
    else:
        output_dir = '{}_finetune:{}_{}_{}M_{}_{}epochs'.format(CURRENT,args.finetune,model_type, args.attention_size, args.decoder_name,args.n_epochs)

    output_dir = os.path.join(args.experiments_dir, output_dir)
    
    

    print(output_dir)
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs, 
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate = args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs, 
        logging_strategy="epoch", 
        output_dir=output_dir, 
        overwrite_output_dir=True, 
    )
    
    if args.extractor=='CLIP':
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator, 
            train_dataset=train_dataset,
            tokenizer=feature_extractor, 
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator, 
            train_dataset=train_dataset,
        )
    trainer.train()
    '''
    trainer의 tokenizer는 optional 
    tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        
    '''
    args_text = str(args)
    with open(os.path.join(output_dir,"args.txt"), "w") as file:
        file.write(args_text)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_dir", type=str, default="features/", help="Directory where cached input image features are stored")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--experiments_dir", type=str, default="/data/minkuk/caption/experiments/", help="Directory where trained models will be saved")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="gpt2, opt // Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=28, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False, help="Whether to train the decoder in addition to the attention")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--disable_prev", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=1, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="/local_datasets/caption/bank/anet/i3d/scene_sentences.txt", help="JSON file with retrieved captions")
    parser.add_argument("--parag_template_path", type=str, default="src/template.txt", help="TXT file with template")
    parser.add_argument("--dvc_template_path", type=str, default="src/dvc_template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")

    parser.add_argument("--ablation_visual", action="store_true", default=False, help="Whether to blank visual features")
    
    ###
    parser.add_argument("--extractor", default='I3D', help="which extractor is used to extract visual features. (TSN, CLIP, I3D)")
    parser.add_argument("--cap_task", type=str, default="dense",help="dense / parag")
    
    parser.add_argument("--base_path", type=str, default="/local_datasets/caption/", help="Directory where all data saved")
    parser.add_argument("--dataset", type=str, default="anet", help="anet, yc2")
    parser.add_argument("--backbone", type=str, default="tsp", help="i3d,tsp,tsn,clip")
    parser.add_argument("--mode", type=str, default="train", help="train,val,test") #in saving bank, only train 
    parser.add_argument("--bank_path",type = str, default = '/local_datasets/caption/bank')
    parser.add_argument("--bank_type", nargs='+', default=['anet'], help="which domain will be used in ret bank // ['anet','yc2','image']")
    #use bank type like this  ##python train.py --bank_type anet yc2 image
    parser.add_argument("--finetune", action="store_true",default=False)
    parser.add_argument("--num_gpu", type=int,default=1)
    parser.add_argument("--featdim", type=int,default=512)
    
    parser.add_argument("--bert_score",action="store_true",default=False,help="Whether you use bert score or not when selecting retrieval caption ")
    args = parser.parse_args()

    args.lr *= args.num_gpu
    args.lr = args.lr * (args.batch_size/32) * args.gradient_steps

    

    main(args)




