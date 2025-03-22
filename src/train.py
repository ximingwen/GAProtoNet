
import torch
import os
import json
import random
import numpy as np
import argparse
import logging
from transformers import BertTokenizer,BertConfig
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import DistilBertTokenizer, DistilBertModel
from models.model import LLM_Baseline, LLM_GraphAttentionPrototype
from datas.data_set import TextClassificationSet,collate_func
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm,trange
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger=logging.getLogger(__name__)

def train(model,device,tokenizer,args):
    '''
    :param model:
    :param device:
    :param tokenizer:
    :param args:
    :return:
    '''
    tb_write=SummaryWriter()
    if args.gradient_accumulation_steps<1:
        raise ValueError('Gradient accumulation parameter is invalid, it must be greater than or equal to 1.')
    train_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
    train_data=TextClassificationSet(tokenizer,args.max_len,args.data_dir,"train",path_file=args.train_file_path)
    train_sampler=RandomSampler(train_data)
    train_data_loader=DataLoader(train_data,
                                 sampler=train_sampler,
                                 batch_size=train_batch_size,
                                 collate_fn=collate_func)
    total_steps=int(len(train_data_loader)*args.num_train_epochs/args.gradient_accumulation_steps)

    dev_data=TextClassificationSet(tokenizer,args.max_len,args.data_dir,"dev",path_file=args.dev_file_path)
    # test_data = SacasamDetection(tokenizer, args.max_len, args.data_dir, "test_sacasam_detection",path_file=args.test_file_path)
    logging.info("The total number of training steps is:{}".format(total_steps))
    model.to(device)
    # Get all model parameters and select the ones for which weight decay is not desired
    param_optimizer=list(model.named_parameters())
    no_decay=["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_grouped_parameters=[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight':0.0}
    ]
    # Set the optimizer
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=args.adam_epsilon)
    #optimizer=AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    # unfreeze_layers = ['cls.']
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             break
   # Validate
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())
    # Filter out parameters with requires_grad = False
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    schedular=get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=int(args.warmup_proportion*total_steps),
                                          num_training_steps=total_steps)
    ce_loss = CrossEntropyLoss()
    # Clear CUDA cache
    torch.cuda.empty_cache()
    model.train()
    tr_loss,logging_loss=0.0,0.0
    best_acc= 0.0
    global_step=0
    for iepoch in trange(0,int(args.num_train_epochs),desc="Epoch",disable=False):
        iter_bar=tqdm(train_data_loader,desc="Iter (loss=X.XXX)",disable=False)
        for step,batch in enumerate(iter_bar):
            input_ids=batch["input_ids"].to(device)
            token_type_ids=batch["token_type_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            position_ids=batch["position_ids"].to(device)
            labels=batch["labels"].to(device)
            outputs = model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  labels=labels)

          
            loss=ce_loss(outputs,labels) + args.prototype_loss_weights['diversity_loss_p_p']*model.prototype_layer.diversity_loss_p_p()
            tr_loss+=loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)"%loss.item())
            # Check if gradient accumulation is applied. If so, divide the loss by the accumulation steps and update parameters every few steps.
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)
            # If the step number is divisible by the accumulation steps, perform parameter optimization.
            if (step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                schedular.step()
                optimizer.zero_grad()
               
                if args.logging_steps>0 and global_step%args.logging_steps==0:
                    tb_write.add_scalar("lr",schedular.get_lr()[0],global_step)
                    tb_write.add_scalar("train_loss",(tr_loss-logging_loss)/(args.logging_steps*args.gradient_accumulation_steps),global_step)
                    logging_loss=tr_loss
        
        global_step+=1
        # If the step number is divisible by save_model_steps, save the trained model.
        if args.save_model_steps>0 and global_step%args.save_model_steps==0:
            eval_acc,f1,precision,recall,json_data=evaluate(model,device,dev_data,args)
            model.train()
            logger.info("dev_acc:{}".format(eval_acc))
            tb_write.add_scalar("dev_acc",eval_acc,global_step)
            output_dir=os.path.join(args.output_dir,"checkpoint-{}".format(global_step))

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # model_to_save=model.module if hasattr(model,"module") else model
            # model_to_save.save_pretrained(output_dir)
            json_output_dir=os.path.join(output_dir,"json_data.json")
            fin=open(json_output_dir,'w',encoding='utf-8')
            fin.write(json.dumps(json_data,ensure_ascii=False,indent=4))
            fin.close()
            # test_acc,test_json_data=evaluate(model,device,test_data,args)
            # model.train()
            # logger.info("test_acc:{}".format(test_acc))
            # tb_write.add_scalar("test_acc",test_acc,global_step)
            # json_output_dir=os.path.join(output_dir,"test_json_data.json")
            # fin=open(json_output_dir,"w",encoding='utf-8')
            # fin.write(json.dumps(test_json_data,ensure_ascii=False,indent=4))
            # fin.close()
            if eval_acc > best_acc:
                if not os.path.exists(args.best_model_dir):
                    os.mkdir(args.best_model_dir)
                torch.save(model, os.path.join(args.best_model_dir,"best_model_xlnet_hotel_multihead_pro20.pth"))
                best_acc = eval_acc
        torch.cuda.empty_cache()
    eval_acc,f1,precision,recall,json_data=evaluate(model,device,dev_data,args)
    logger.info("dev_acc:{}".format(eval_acc))
    tb_write.add_scalar("dev_acc",eval_acc,global_step)
    output_dir=os.path.join(args.output_dir,"checkpoint-{}".format(global_step))
    logger.info("best_acc:{}".format(best_acc))
    tb_write.add_scalar("best_acc",best_acc,global_step)
    output_dir=os.path.join(args.output_dir,"best-{}".format(global_step))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # model_to_save = model.module if hasattr(model, "module") else model
    # model_to_save.save_pretrained(output_dir)
    json_output_dir = os.path.join(output_dir, "json_data.json")
    fin = open(json_output_dir, 'w', encoding='utf-8')
    fin.write(json.dumps(json_data, ensure_ascii=False, indent=4))
    fin.close()
    # test_acc, test_json_data = evaluate(model, device, dev_data, args)
    # model.train()
    # logger.info("test_acc:{}".format(test_acc))
    # tb_write.add_scalar("test_acc", test_acc, global_step)
    # json_output_dir = os.path.join(output_dir, "test_json_data.json")
    # fin = open(json_output_dir, "w", encoding='utf-8')
    # fin.write(json.dumps(test_json_data, ensure_ascii=False, indent=4))
    # fin.close()

def evaluate(model,device,dev_data,args):
    '''
    :param model:
    :param device:
    :param dev_data:
    :param args:
    :return:
    '''
    test_sampler=SequentialSampler(dev_data)
    test_data_loader=DataLoader(dev_data,sampler=test_sampler,batch_size=args.test_batch_size,collate_fn=collate_func)
    iter_bar=tqdm(test_data_loader,desc="iter",disable=False)
    y_true=[]
    y_predict=[]
    y_scores=[]
    samples=[]
    for step,batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            labels=batch["labels"]
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            prediction = model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  labels=labels)
            # scores,prediction=model.forward(input_ids=input_ids,
            #                       token_type_ids=token_type_ids)
            y_true.extend(labels.numpy().tolist())
           
            y_predict.extend( np.argmax(prediction.cpu().numpy(), axis=1).tolist())
            # y_scores.extend(scores.cpu().numpy().tolist())
            samples.extend(batch["samples"])
    json_data={"data":[],"acc":None}
    for label,pre,score,sample in zip(y_true,y_predict,y_scores,samples):
        sample["label"]=label
        sample["pre"]=pre
        # sample["scores"]=score
        json_data["data"].append(sample)
    y_true=np.array(y_true)
    y_predict=np.array(y_predict)
    eval_acc=np.mean((y_true==y_predict))
    f1 = f1_score(y_true, y_predict, average='binary')
    precision = precision_score(y_true, y_predict, average='binary')
    recall = recall_score(y_true, y_predict, average='binary')  
    
    json_data["acc"]=str(eval_acc)
    return eval_acc,f1,precision,recall,json_data

def set_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='0',type=str,help='Set the GPU to be used during training or testing.')
    # parser.add_argument('--train_file_path',default='./SemEval2022/train/train.En.csv',type=str,help='Training data')
    # parser.add_argument('--dev_file_path', default='./SemEval2022/dev/dev.En.csv', type=str, help='Validation data')
    # parser.add_argument('--test_file_path', default='./SemEval2022/test/task_A_En_test.csv', type=str, help='Testing data')
    parser.add_argument('--train_file_path',default="/root/autodl-fs/hotel/train_data.csv",type=str,help='Training data')
    parser.add_argument('--dev_file_path', default="/root/autodl-fs/hotel/test_data.csv", type=str, help='Validation data')
    # parser.add_argument('--test_file_path', default='./SemEval2022/test/task_A_En_test.csv', type=str, help='Testing data')
    parser.add_argument('--data_dir', default='./cached/', type=str, help='Path to store cached data')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='Number of epochs for model training')
    parser.add_argument('--train_batch_size', default=256, type=int, help='Batch size for each training step')
    parser.add_argument('--test_batch_size', default=64, type=int, help='Batch size for each testing step')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Warmup ratio, i.e., the percentage of the total training steps used for warmup.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='The epsilon value of the Adam optimizer.')
    parser.add_argument('--save_model_steps', default=5, type=int, help='The number of steps to save the trained model.')
    parser.add_argument('--logging_steps', default=300, type=int, help='The number of steps after which to save the training logs.')
    parser.add_argument('--gradient_accumulation_steps', default=64, type=int, help='Gradient accumulation')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='baselinep/', type=str, help='Model output path')
    parser.add_argument('--best_model_dir', default='/root/autodl-fs/model', type=str, help='Best model output path')
    parser.add_argument('--seed', default=2022, type=int, help='Random seed')
    parser.add_argument('--max_len', default=512, type=int, help='Maximum length of the input text for the model')
    
    # 模型参数
    parser.add_argument('--bert_model_path', default='/root/autodl-fs/xlnet/', type=str, help='Path to the pretrained model')
    parser.add_argument('--bert_config', default='/root/autodl-fs/xlnet/config.json', type=str, help='Pretrained model configuration file')
    parser.add_argument('--llm_model_path', default='./root/autodl-fs/xlnet/model.pt', type=str, help='Pretrained model path')
    parser.add_argument('--frozen_layers', default=16, type=int, help='Number of frozen layers in the pretrained model')
    parser.add_argument('--vocab_path', default='/root/autodl-fs/xlnet/vocab.json', type=str, help='Pretrained model vocabulary data')
    parser.add_argument('--merge_path', default='/root/autodl-fs/xlnet/merges.txt', type=str, help='Pretrained model merge file')
    parser.add_argument('--tokenizer_path', default='/root/autodl-fs/xlnet/tokenizer.json', type=str, help='Pretrained model tokenizer file')
    parser.add_argument('--tokenizer_config', default='/root/autodl-fs/xlnet/tokenizer_config.json', type=str, help='Pretrained model tokenizer configuration file')
    parser.add_argument('--bert_cls_dim', default=1024, type=int, help='Output dimension of the BERT model')
    parser.add_argument('--prototype_dim', default=1024, type=int, help='Output dimension of the BERT model')
    parser.add_argument('--attention_dim', default=4096, type=int, help='attention')
    parser.add_argument('--q_dim', default=2048, type=int, help='Dimension of the prototype layer')
    parser.add_argument('--k_dim', default=2048, type=int, help='Output dimension of the BERT model')
    parser.add_argument('--v_dim', default=4096, type=int, help='Dimension of the prototype layer')
    parser.add_argument('--num_prototypes', default=20, type=int, help='Number of prototypes')
    parser.add_argument('--prototype_threshold', default=0.5, type=float, help='Prototype threshold')
    parser.add_argument('--prototype_loss_weights', default='{"diversity_loss_z_p": 0.1, "diversity_loss_p_z": 0.1, "diversity_loss_p_p": 0.0, "loss_num_prototypes": 0.1, "cluster_loss": 0.1, "seperation_loss": 0.1}', type=json.loads, help='String representation of the prototype loss weight dictionary')
    parser.add_argument('--transformer_dim', default=256, type=int, help='Dimension of the Transformer layer')
    parser.add_argument('--transformer_layers', default=2, type=int, help='Number of Transformer layers')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='Transformer的dropout rate')
    parser.add_argument('--fc_output_dim', default=4096, type=int, help='Output dimension of the fully connected layer')
    parser.add_argument('--mlp_hidden_dim', default=[1024,256,64], type=list, help='Hidden layer dimensions of the MLP')
    parser.add_argument('--mlp_output_dim', default=2, type=int, help='Output dimension of the MLP')
    parser.add_argument('--mlp_num_layers', default=2, type=int, help='Number of layers in the MLP')
    parser.add_argument('--mlp_dropout', default=0.1, type=float, help='Dropout rate of the MLP')
    parser.add_argument("--normalization", default="sparsemax", type=str)
    parser.add_argument("--num_heads", default=1, type=int, help='Graph Attention')
    parser.add_argument("--alpha", default=0.2, type=float, help='Graph Attention LeakyRelu')
    parser.add_argument("--hierarchical_prototype_num_layer", default=3, type=int, help='Hierarchical Prototype Layer')
    parser.add_argument("--G_dim", default=32, type=int, help='k')
    parser.add_argument("--gru_hidden_dim", default=512, type=int, help='Gru Hidden Dim')
    parser.add_argument("--gru_num_layer", default=1, type=int, help='Gru Num Layer')
    return parser.parse_args() 

def main():
    args=set_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device=torch.device("cuda" if torch.cuda.is_available() and int(args.device)>=0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # model_path='pre_train_model/sci-uncased/pytorch_model.bin'
    # config_path='pre_train_model/sci-uncased/config.json'
    # state_dict=torch.load(model_path,map_location='cpu')
    # a=state_dict['bert.embeddings.token_type_embedding.weight'].tolist()
    # b = state_dict['bert.embeddings.token_type_embedding.weight'].tolist()
    # c=a+b
    # state_dict['bert.embeddings.token_type_embedding.weight']=torch.tensor(c)
    # model_config=BertConfig.from_json_file(config_path)
    # model=BertTorchClassfication.from_pretrained(None,config=config_path,state_dict=state_dict)
    # model=BertTorchClassfication.from_pretrained(args.pretrained_model_path)
    
    #实例化tokenizer
    #tokenizer=BertTokenizer.from_pretrained(args.vocab_path,do_lower_case=True)
    #tokenizer=BertTokenizer.from_pretrained("bert-large-uncased",do_lower_case=True)
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-large',do_lower_case=True)
    '''
    tokenizer = RobertaTokenizer.from_pretrained( 
        pretrained_model_name_or_path = args.bert_model_path,
        vocab_file=args.vocab_path,  # Usually, there is no need to directly specify the vocab_file unless there is a specific instruction in the tokenizer_config  
        merges_file=args.merge_path,  # For some tokenizers (like BPE), you may need to specify the merges file  
        tokenizer_file=args.tokenizer_path,  # Your tokenizer file  
        tokenizer_config_file=args.tokenizer_config,  # Your tokenizer configuration file  
        cache_dir=None,  # Optional, specify the cache directory  
        force_download=False,  # Force download, typically not needed  
        resume_download=False,  # Resume download, typically not needed  
        proxies=None,  # Proxy settings, usually not required  
        use_fast=True,  # Try to use the fast tokenizer (if available)   
    )
    for i in range(1,100):
        tokenizer.add_tokens("[uncased{}]".format(i), special_tokens=True)
    # Create the output directory for the model
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    '''  
       
    tokenizer = XLNetTokenizer.from_pretrained( 
        pretrained_model_name_or_path = args.bert_model_path ,
        tokenizer_file=args.tokenizer_path    
        )
    for i in range(1,100):
        tokenizer.add_tokens("[uncased{}]".format(i),special_tokens=True)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    '''
    tokenizer = DistilBertTokenizer.from_pretrained( 
        pretrained_model_name_or_path = args.bert_model_path,
        vocab_file=args.vocab_path,  # Usually, there is no need to directly specify the vocab_file unless there is a specific instruction in the tokenizer_config  
        #merges_file=args.merge_path,  # For some tokenizers (like BPE), you may need to specify the merges file  
        tokenizer_file=args.tokenizer_path,  # Your tokenizer file  
        tokenizer_config_file=args.tokenizer_config,  # Your tokenizer configuration file  
        cache_dir=None,  # Optional, specify the cache directory  
        force_download=False,  # Force download, typically not needed  
        resume_download=False,  # Resume download, typically not needed  
        proxies=None,  # Proxy settings, usually not required  
        use_fast=True,  # Try to use the fast tokenizer (if available)   
    )
    '''
    '''
    model = LLM_Baseline(args.bert_model_path, args.bert_config, args.frozen_layers, args.bert_cls_dim, args.attention_dim, args.prototype_dim, args.num_prototypes, args.prototype_threshold, args.prototype_loss_weights, args.transformer_dim, args.transformer_layers, args.num_heads, args.transformer_dropout, args.fc_output_dim, args.mlp_hidden_dim, args.mlp_output_dim, args.mlp_num_layers, args.mlp_dropout, args.normalization)
    '''
       
    model = LLM_GraphAttentionPrototype(args.bert_model_path, args.bert_config, args.frozen_layers, args.bert_cls_dim, args.attention_dim, args.prototype_dim, args.num_prototypes, args.prototype_threshold, args.prototype_loss_weights, args.transformer_dim, args.transformer_layers, args.num_heads, args.transformer_dropout, args.fc_output_dim, args.mlp_hidden_dim, args.mlp_output_dim, args.mlp_num_layers, args.mlp_dropout, args.normalization,args.k_dim,args.q_dim,args.v_dim)
    
    train(model,device,tokenizer,args)
    

if __name__=="__main__":
    main()
