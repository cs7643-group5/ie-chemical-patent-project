from torch import nn
import torch
import pdb
import numpy as np

def make_emask(sentences,trig_mask,ent_mask,batch_size):

    max_seq = sentences.shape[1]
    #e_mask1 = np.zeros((batch_size,max_seq))
    #e_mask2 = np.zeros((batch_size,max_seq))
    #sentences,_,_, trig_mask, ent_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
    for i in range(batch_size):
        #print(sentences.shape)
        e_mask1 = np.zeros((batch_size,max_seq))
        e_mask2 = np.zeros((batch_size,max_seq))
        #print(e_mask1.shape)
        trig_idx_1 = trig_mask[i][0]
        trig_idx_2 = trig_mask[i][1]
        e_mask1[i,trig_idx_1:trig_idx_2] = 1
        ent_idx_1 = ent_mask[i][0]
        ent_idx_2 = ent_mask[i][1]
        e_mask2[i,ent_idx_1:ent_idx_2] = 1
    #print(e_mask1)
    #print(e_mask1.shape)
    e_mask1 = torch.tensor(e_mask1)
    #print("Tensor")
    #print(e_mask1)
    #print(e_mask1.shape)
    e_mask2 = torch.tensor(e_mask2)
    return e_mask1, e_mask2
    
def extract_entity(sequence_output, e_mask):
    #print("haha")
    #print(e_mask.shape)
    extended_e_mask = e_mask.unsqueeze(1)
    #print(extended_e_mask.shape)
    #print(sequence_output.shape)
    extended_e_mask = torch.bmm(
        extended_e_mask.float(), sequence_output).squeeze(1)
    #return extended_e_mask.float()
    return extended_e_mask.float()

def entity_average(hidden_output, e_mask):
    #e_mask.to(device)
    e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    #print("emask unsqueeze")
    #print(e_mask_unsqueeze)
    #print("float")
    #print(e_mask_unsqueeze.float())
    #length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
    #print("length")
    #print(length_tensor)
    #print("hidden")
    #print(hidden_output)
    # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)

    #print("sum")
    #print(sum_vector)
    avg_vector = sum_vector.float() 
    return avg_vector


class RelationClassifier(nn.Module):
    def __init__(self,
                 bert_encoder: nn.Module,
                 enc_hid_dim=768,  # default embedding size
                 outputs=3,
                 dropout=0.1):
        super().__init__()

        self.bert_encoder = bert_encoder

        self.enc_hid_dim = enc_hid_dim
        self.e1_linear = nn.Linear(enc_hid_dim, enc_hid_dim)
        self.tanh = nn.Tanh()
        self.e2_linear = nn.Linear(enc_hid_dim, enc_hid_dim)
        self.linear_1_class = nn.Linear(in_features=enc_hid_dim*3, out_features=enc_hid_dim*3)
        self.relu_class = nn.ReLU()
        self.linear_2_class = nn.Linear(in_features=enc_hid_dim*3, out_features=outputs)
        self.dropout_layer_class = nn.Dropout(p=dropout)

    def forward(self,
                src,
                mask,
                trig_mask,
                ent_mask):

        bert_output = self.bert_encoder(src, mask)

        # bert sequence classifier
        sequence_output = bert_output.last_hidden_state
        cls_output = bert_output.pooler_output

        #trig_output = []
        #ent_output = []
        #for i, (t, e) in enumerate(zip(trig_mask, ent_mask)):
        #    trig_output.append(sequence_output[i, t[0], :])
        #    ent_output.append(sequence_output[i, e[0], :])
        e1_mask,e2_mask = make_emask(src,trig_mask,ent_mask,sequence_output.shape[0])
        #print("hehe")
        #print(sequence_output.shape)
        #e2_mask = make_emask(src,trig_mask,ent_mask,sentences.shape[0])
        if torch.cuda.is_available():
            #print('using cuda')
            device = torch.device('cuda')
        else:
            #print('using cpu')
            device = torch.device('cpu')

        e1_h = entity_average(sequence_output.to(device), e1_mask.to(device))
        e2_h = entity_average(sequence_output.to(device), e2_mask.to(device))
        
        e1_h = self.e1_linear(e1_h)
        e1_h = self.tanh(e1_h)
        e2_h = self.e2_linear(e2_h)
        e2_h = self.tanh(e2_h)
        #print("e1")
        #print(e1_h.shape)
        #print(e1_h)
        #trig_output = torch.stack(trig_output, dim=0)
        #ent_output = torch.stack(ent_output, dim=0)

        #combined_output = torch.cat((cls_output, trig_output, ent_output), axis=1)
        #print("cls")
        #print(cls_output.shape)
        #e1_h = 
        combined_output = torch.cat((cls_output, e1_h, e2_h), axis=1)
        #print("combined")
        #print(combined_output.shape)
        # apply simple feed forward network for classification
        u_1 = self.linear_1_class(combined_output)
        o_1 = self.relu_class(u_1)
        o_1_drop = self.dropout_layer_class(o_1)
        scores = self.linear_2_class(o_1_drop)

        return scores
