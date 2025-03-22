
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence
logger=logging.getLogger(__name__)

class TextClassificationSet(Dataset):
    def __init__(self,tokenizer,max_len,data_dir,data_set_name,path_file=None,is_over_write=True):
        '''
        Initialization function
        :param tokenizer: The tokenizer
        :param max_len: The maximum length of the data
        :param data_dir: The path to save the cached files
        :param data_set_name: The name of the dataset
        :param path_file: The raw data file
        :param is_over_write: Whether to regenerate the cache file
        :return:
        '''
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.data_set_name=data_set_name
        cached_feature_file=os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        # Check if the cached file exists. If it does, load the processed data directly.
        if os.path.exists(cached_feature_file) and not is_over_write:
            logger.info('The cached file {} already exists, loading directly.'.format(cached_feature_file))
            self.data_set=torch.load(cached_feature_file)["data_set"]
        # If the cached file does not exist, process the raw data and save the processed data as a cached file.
        else:
            logger.info('The cached file {} does not exist, performing data preprocessing.'.format(cached_feature_file))
            self.data_set=self.load_data(path_file)
            logger.info('Data preprocessing is complete. The processed data is saved to {} as the cached file.'.format(cached_feature_file))
            torch.save({'data_set':self.data_set},cached_feature_file)

    def load_data(self,path_file):
        '''
        Load the raw data and generate the processed data
        :param path_file: Path to the raw data
        :return:
        '''
        self.data_set=[]
      
        # df = pd.read_csv(path_file, encoding='utf-8-sig')
        # df = pd.read_csv(path_file, delimiter='\t')
        df = pd.read_csv(path_file)
        for index, row in df.iterrows():
            sentence=row['text']
            label = int(row["label"])
            # if int(row['label']) ==0:
                
            #     label = [1,0]
            # else:
            #     label = [0,1]

            
            try: 
                input_ids,token_type_ids,position_ids,attention_mask=self.convert_feature(sentence)
                # if len(input_ids)>500:
                #     continue
                self.data_set.append({"text":sentence,
                                    "input_ids":input_ids,
                                    "token_type_ids":token_type_ids,
                                    "attention_mask":attention_mask,
                                    "position_ids":position_ids,
                                    "label":label})
            except:
                pass
        return self.data_set

    def convert_feature(self,sentence):
        '''
        Data processing function
        :param sample: The input text for each sample
        :return:
        '''
        sentence_tokens = self.tokenizer.tokenize(sentence)
        input_ids=self.tokenizer.convert_tokens_to_ids(sentence_tokens)
        token_type_ids=[0]*len(input_ids)
        position_ids=[s for s in range(len(input_ids))]
        attention_mask=[1]*len(input_ids)
        assert len(input_ids)==len(token_type_ids)
        assert len(input_ids)==len(attention_mask)
        assert len(input_ids)<=512
        return input_ids,token_type_ids,position_ids,attention_mask

    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, idx):
        instance=self.data_set[idx]
        return instance

def collate_func(batch_data):
    '''
    The collate_func function required by DataLoader, processes data into tensor format
    :param batch_data: Batch data
    :return:
    '''
    batch_size=len(batch_data)
    # If batch_size is 0, return an empty dictionary.
    if batch_size==0:
        return {}
    input_ids_list,token_type_ids_list,position_ids_list,attention_mask_list,label_list=[],[],[],[],[]
    sample_list=[]
    for instance in batch_data:
        # Pad the data according to the maximum data length in the batch.
        input_ids_temp=instance["input_ids"]
        token_type_ids_temp=instance["token_type_ids"]
        position_ids_temp=instance["position_ids"]
        attention_mask_temp=instance["attention_mask"]
        label_temp=instance["label"]
        sample={"text":instance["text"]}
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))
        position_ids_list.append(torch.tensor(position_ids_temp,dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp,dtype=torch.long))
        label_list.append(label_temp)
        sample_list.append(sample)
    # Pad the length of all tensors in the list.
    return {"input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=0),
            "position_ids":pad_sequence(position_ids_list,batch_first=True,padding_value=0),
            "attention_mask":pad_sequence(attention_mask_list,batch_first=True,padding_value=0),
            "labels":torch.tensor(label_list,dtype=torch.long),
            "samples":sample_list}




