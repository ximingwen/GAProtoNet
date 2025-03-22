
from torch import nn
import torch
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel,BertSelfAttention
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
import logging
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F

logger=logging.getLogger(__name__)

        
class LargLanguageModel(nn.Module):
    '''
    LargLanguageModel
    '''
    base_model_prefix = "llm"

    def __init__(self,llm_model_path,config,bert_cls_dim,attention_dim):
        super(LargLanguageModel, self).__init__()
        #self.llm=RobertaModel.from_pretrained(llm_model_path, config=config)
        #self.llm = torch.load(llm_model_path)
        self.llm = XLNetModel.from_pretrained(llm_model_path, config=config)

        self.encoder_key = nn.Linear(in_features=bert_cls_dim, out_features=attention_dim)
        self.encoder_query = nn.Linear(in_features=bert_cls_dim, out_features=attention_dim)
        self.encoder_value = nn.Linear(in_features=bert_cls_dim, out_features=attention_dim)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,position_ids=None,
                index1=None,index2=None):
        '''
        :param input_ids: Convert the input text to token_ids using the tokenizer
        :param attention_mask: A mask vector consisting of 0s and 1s, indicating which part the model should focus on
        :param token_type_ids: Segment vector
        :param labels: Labels
        :param position_ids: Position information vector
        :param index1: Position information of the first text for attention
        :param index2: Position information of the second text for attention
        :return:
        '''
        outputs=self.llm.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids)
        cls_representation = outputs.last_hidden_state[:, 0, :]

        encoded_query = self.encoder_query(cls_representation)
        encoded_value = self.encoder_value(cls_representation)

        # encoded_query =  cls_representation
        # encoded_value =  cls_representation

        '''
        # If attention is required, use this part, and modify the linear regression parameter config.hidden_size to the corresponding size * n
        all_data = outputs[0]  # Output of the entire sentence
        attention_1 = torch.index_select(all_data, dim=1, index=index1)
        attention_2 = torch.index_select(all_data, dim=1, index=index2)
        extra_attention_output = self.bert_attention.forward(hidden_states=attention_1,
                                                     encoder_hidden_states=attention_2,
                                                     attention_mask=attention_mask)[0]
        extra_attention = torch.mean(extra_attention_output.float(), dim=1)  # Take the average of the obtained attention
        output = torch.cat((pooled_output, extra_attention), dim=1)  # Concatenate the tensors
        '''
        return cls_representation, encoded_query, encoded_value
    
    
class PrototypeLayer(nn.Module):
    def __init__(self, input_dim, prototype_dim, attention_dim, num_prototypes,threshold, loss_weights):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_positive_prototypes = num_prototypes // 2  # Number of positive prototypes
        self.num_negative_prototypes = num_prototypes - self.num_positive_prototypes  # Number of negative prototypes
        self.positive_prototype_indices = torch.arange(self.num_positive_prototypes)  
        self.negative_prototype_indices = torch.arange(self.num_positive_prototypes, self.num_prototypes)
        self.prototype_vectors = nn.Parameter(torch.randn(num_prototypes, prototype_dim))  
        self.linear = nn.Linear(input_dim,prototype_dim)
        self.threshold = threshold
        self.loss_weights = loss_weights 

        self.encoder_key = nn.Linear(in_features=prototype_dim, out_features=attention_dim)
        self.encoder_query = nn.Linear(in_features=prototype_dim, out_features=attention_dim)
        self.encoder_value = nn.Linear(in_features=prototype_dim, out_features=attention_dim)
  

    def forward(self, x, labels):

        encoded_key = self.encoder_key(self.prototype_vectors)
        encoded_value = self.encoder_value(self.prototype_vectors)
        # encoded_key = self.prototype_vectors
        # print(self.prototype_vectors.shape)
        # print(encoded_key.shape)
        # z = self.linear(x)
        # cosine_similarities = F.cosine_similarity(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), dim=-1)
        # token_sequences = cosine_similarities.unsqueeze(2) * self.prototype_vectors.unsqueeze(0)
        # loss = self.loss_weights['diversity_loss_z_p'] * self.diversity_loss_z_p(z) + self.loss_weights['diversity_loss_p_z'] * self.diversity_loss_p_z(z) + self.loss_weights['diversity_loss_p_p'] * self.diversity_loss_p_p() + self.loss_weights['loss_num_prototypes'] * self.loss_num_prototypes(z) + self.loss_weights['cluster_loss'] * self.cluster_loss(z, labels, cosine_similarities) + self.loss_weights['seperation_loss'] * self.cluster_loss(z, labels, cosine_similarities)
        # return token_sequences, loss

        return encoded_key, encoded_value
 
    def diversity_loss_z_p(self,z):
        distances = torch.cdist(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), p=2)
        nearest_indices = torch.argmin(distances, dim=0)
        nearest_samples = z[nearest_indices]
        distances_per_prototype = torch.norm(nearest_samples - self.prototype_vectors, dim=-1)
        loss = distances_per_prototype.mean()
        return loss
    
    def diversity_loss_p_z(self,z):
        distances = torch.cdist(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), p=2)
        nearest_indices = torch.argmin(distances, dim=1)
        nearest_prototypes = self.prototype_vectors[nearest_indices]
        distances_per_sample = torch.norm(z - nearest_prototypes, dim=-1)
        loss = distances_per_sample.mean()
        return loss
    
    def diversity_loss_p_p(self):
        distances = torch.cdist(self.prototype_vectors, self.prototype_vectors, p=2)
        device = torch.device('cuda:0')
        mask = torch.triu(torch.ones(self.num_prototypes,self.num_prototypes),diagonal=0).to(device)
        distances = torch.mul(distances,mask)
        average_distance = distances.sum()/mask.sum()
        return -average_distance
    
    def loss_num_prototypes(self,z):
        similarities = F.cosine_similarity(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), dim=-1)
        num_similar_prototypes = (similarities > self.threshold).sum(dim=1)
        penalize_prototypes = (num_similar_prototypes > self.num_prototypes / 4).float()
        penalty_loss = penalize_prototypes.sum()
        return penalty_loss
    
    def cluster_loss(self, z, labels, cosine_similarities):
        positive_indices = torch.nonzero(labels == 1, as_tuple=True)[0]
        positive_prototype_similarities = cosine_similarities[positive_indices[:, None], self.positive_prototype_indices]
        min_positive_prototype_similarities, _ = positive_prototype_similarities.max(dim=1)
        positive_distances = 1 - min_positive_prototype_similarities
        negative_indices = torch.nonzero(labels == 0, as_tuple=True)[0] 
        negative_prototype_similarities = cosine_similarities[negative_indices[:, None], self.negative_prototype_indices]
        min_negative_prototype_similarities, _ = negative_prototype_similarities.max(dim=1)
        negative_distances = 1 - min_negative_prototype_similarities
        total_distances = torch.cat([positive_distances, negative_distances], dim=0)
        average_loss = total_distances.mean()  
        return average_loss
    
    def seperation_loss(self, z, labels, cosine_similarities):
        positive_indices = torch.nonzero(labels == 1, as_tuple=True)[0]
        positive_prototype_similarities = cosine_similarities[positive_indices[:, None], self.negative_prototype_indices]
        min_positive_prototype_similarities, _ = positive_prototype_similarities.max(dim=1)
        positive_distances = min_positive_prototype_similarities
        negative_indices = torch.nonzero(labels == 0, as_tuple=True)[0]
        negative_prototype_similarities = cosine_similarities[negative_indices[:, None], self.positive_prototype_indices]
        min_negative_prototype_similarities, _ = negative_prototype_similarities.max(dim=1)
        negative_distances = min_negative_prototype_similarities
        total_distances = torch.cat([positive_distances, negative_distances], dim=0)
        average_loss = total_distances.mean()  
        return average_loss        
    

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))  
        self.layers.append(nn.ReLU())  
           
        for i in range(len(hidden_dims) - 1):  
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))  
            self.layers.append(nn.ReLU())  
            self.layers.append(nn.Dropout(dropout))   
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
      
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LLM_GraphAttentionPrototype(nn.Module):
    def __init__(self, llm_model_path, config, frozen_layers, bert_cls_dim, attention_dim, prototype_dim, num_prototypes, prototype_threshold, prototype_loss_weights, transformer_dim, transformer_layers, num_heads, transformer_dropout, fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout, normalization,k_dim,q_dim,v_dim):
        super(LLM_GraphAttentionPrototype, self).__init__()
        #self.bert_cls = BertTorchClassfication(bert_cls_dim, bert_model_path, config)
        self.llm = LargLanguageModel(llm_model_path, config,bert_cls_dim, attention_dim)
        for name, param in self.llm.named_parameters():  
            if 'llm.layer' in name and int(name.split('.')[2]) < frozen_layers:  
                param.requires_grad = False
        self.prototype_layer = PrototypeLayer(bert_cls_dim, prototype_dim,attention_dim, num_prototypes, prototype_threshold, prototype_loss_weights)
        # self.transformer_layer = TransformerLayer(prototype_dim, transformer_layers, transformer_heads, transformer_dim, transformer_dropout)
        self.fc_layer = FullyConnectedLayer(num_prototypes, mlp_output_dim, mlp_dropout)
        self.mlp = MLP(fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout)
        self.attention_dim = attention_dim
        self.normalization = normalization
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.device = 'cuda:0'
        
        self.encoder_key = [nn.Linear(in_features=prototype_dim, out_features=k_dim) for i in range(num_heads)]
        self.encoder_query = [nn.Linear(in_features=bert_cls_dim, out_features=q_dim) for i in range(num_heads)]
        self.encoder_value = [nn.Linear(in_features=prototype_dim, out_features=v_dim) for i in range(num_heads)]
        self.fc = nn.Linear(in_features=num_heads*v_dim, out_features=v_dim)
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        cls_token, __, __  = self.llm( input_ids, attention_mask, token_type_ids, labels)
        multi_graph_encoded_cls = []
               
        for i in range(self.num_heads):
            # encoded q, k ,v
            encoded_queries = self.encoder_query[i].to(self.device)(cls_token)
            encoded_proto_keys  = self.encoder_key[i].to(self.device)(self.prototype_layer.prototype_vectors)
            encoded_proto_values = self.encoder_value[i].to(self.device)(self.prototype_layer.prototype_vectors)

            # Matrix multiplication
            activations = torch.matmul(encoded_proto_keys, encoded_queries.transpose(0, 1))
            # Scaling
            activations /= np.sqrt(self.k_dim)
            # Transposing
            activations = activations.transpose(0, 1)

            # consctruct graph edges
            edge_weight_coefs = torch.sigmoid(activations)
            #print(weight_coefs)
            edge_weight_coefs = torch.where(torch.abs(edge_weight_coefs) > 0.1, edge_weight_coefs, torch.zeros_like(activations))
            edge_weight_coefs = edge_weight_coefs/torch.abs(edge_weight_coefs).sum()
            num_nonzero_elements = torch.sum(edge_weight_coefs != 0, dim=1, keepdim=True)  

            #print(f"Number of non-zero elements: {num_nonzero_elements}")

            # calculate graph_encoded_cls from neighbours
            graph_encoded_cls = torch.matmul(edge_weight_coefs, encoded_proto_values)
            multi_graph_encoded_cls.append(graph_encoded_cls)
        
        fused_graph_encoded_cls = torch.concat(multi_graph_encoded_cls,dim = 1)
        fused_graph_encoded_cls = self.fc(fused_graph_encoded_cls)
        mlp_output = self.mlp( fused_graph_encoded_cls )
        
        return mlp_output
    
    
class LLM_Baseline(nn.Module):
    def __init__(self, llm_model_path, config, frozen_layers, bert_cls_dim, attention_dim, prototype_dim, num_prototypes, prototype_threshold, prototype_loss_weights, transformer_dim, transformer_layers, transformer_heads, transformer_dropout, fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout, normalization):
        super(LLM_Baseline, self).__init__()
        self.llm = LargLanguageModel(llm_model_path, config,bert_cls_dim, attention_dim)
        for name, param in self.llm.named_parameters():
            if 'llm.layer' in name and int(name.split('.')[2]) < frozen_layers:   
                param.requires_grad = False
        self.prototype_layer = PrototypeLayer(bert_cls_dim, prototype_dim,attention_dim, num_prototypes, prototype_threshold, prototype_loss_weights)
        self.fc_layer = FullyConnectedLayer(num_prototypes, mlp_output_dim, mlp_dropout)
        self.mlp = MLP(fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout)
        self.attention_dim = attention_dim
        self.normalization = normalization

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        cls_token, __, __  = self.llm( input_ids, attention_mask, token_type_ids, labels)
       
        # fc_output = self.fc_layer(transformer_cls_token)
        mlp_output = self.mlp( cls_token )

        #mlp_output = self.fc_layer(weight_coefs )
        return mlp_output
        
