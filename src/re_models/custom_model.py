from torch import nn
import torch
import pdb

class RelationClassifier(nn.Module):
    def __init__(self,
                 bert_encoder: nn.Module,
                 enc_hid_dim=768,  # default embedding size
                 outputs=3,
                 dropout=0.1):
        super().__init__()

        self.bert_encoder = bert_encoder

        self.enc_hid_dim = enc_hid_dim

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

        trig_output = []
        ent_output = []
        for i, (t, e) in enumerate(zip(trig_mask, ent_mask)):
            trig_output.append(sequence_output[i, t[0], :])
            ent_output.append(sequence_output[i, e[0], :])

        trig_output = torch.stack(trig_output, dim=0)
        ent_output = torch.stack(ent_output, dim=0)

        combined_output = torch.cat((cls_output, trig_output, ent_output), axis=1)
        # apply simple feed forward network for classification
        u_1 = self.linear_1_class(combined_output)
        o_1 = self.relu_class(u_1)
        o_1_drop = self.dropout_layer_class(o_1)
        scores = self.linear_2_class(o_1_drop)

        return scores
