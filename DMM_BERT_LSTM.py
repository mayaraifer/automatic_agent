import torch
import torch.nn as nn
import joblib
import random
import numpy as np
import pandas as pd
import copy
from transformers import AdamW
from sklearn import metrics
from create_bert_embadding import load_data_and_make_label_sentence, check_max_len,make_tokenization
import time
from cross_val import cross_vali
from transformers import BertTokenizer,BertModel
epoch_num = 20
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_type = 'blogic'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


class BertModel_():

    def __init__(self):
        super(BertModel_, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)#, output_attentions = False,encoder_hidden_states=True)

class BertLSTM(nn.Module):

    def __init__(self, input_dim,hidden_dim, tagset_size,dropout):
        super(BertLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.bert = BertModel_().bert_model
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, sentence_emb,input_ids,attention_masks):
        whole_emb = torch.zeros((int(input_ids.shape[0]/10),10,768)).to(device)
        for batch in range(0,int(input_ids.shape[0]/10)):
            for i in range(0,10):
                bert_emb = self.bert(input_ids[i+batch*10,:].reshape(1,146),token_type_ids=None,attention_mask=attention_masks[i+batch*10,:].reshape(1,146))
                whole_emb[batch,i,:] = bert_emb['pooler_output'].reshape(768)
        whole_emb = self.drop(whole_emb)
        sentence_emb = torch.cat([self.tanh(sentence_emb), whole_emb], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space


def crete_random_batches(train_df, batch_size):
    training_data,batch, count= [],[],0
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    for index, row in train_df.iterrows():
        game_features = [row[col] for col in list(train_df.columns) if 'features_round' in col]
        text_features = [row[col] for col in list(train_df.columns) if 'review_round' in col]
        labels = row['labels']
        batch.append((game_features,text_features, labels))
        count += 1
        if count == batch_size or index==len(train_df)-1:
            training_data.append(batch)
            batch = []
            count = 0
    return training_data

def training(fold,data,batch_size,hidden_dim,dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()
    model = BertLSTM(input_dim=788, hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model.to(device)
    train_df = data[data['fold']!=fold]
    test = data[data['fold']==fold]
    epoch_counter, min_loss=0,100
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)#, momentum=0.9)#optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epoch_num):
        if epoch==8:
            for n, p in model.named_parameters():
                if 'bert' in n:
                    p.requires_grad = True
            optimizer = AdamW(model.bert.parameters(), lr=1e-5)
        tot_loss =0
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size)
        i=0
        for batch in training_data:
            sentence_game = [sample[0] for sample in batch]
            sentence_text = [sample[1] for sample in batch]
            tags_ = [sample[2] for sample in batch]
            input_ids, attention_masks = make_tokenization(sentence_text, 146, tokenizer,False)
            real_teg = torch.Tensor(tags_).to(device)
            sentence_game = torch.Tensor(sentence_game).to(device)
            attention_masks = attention_masks.to(device)
            input_ids = input_ids.to(device)
            model.zero_grad()
            tag_scores = model(sentence_game,input_ids,attention_masks)
            loss = loss_function(tag_scores.reshape(tag_scores.shape[0],10), real_teg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            tot_loss+=loss.item()
            i+=1
        print('training loss:',tot_loss/len(training_data),type(optimizer))

def main():
    bath_sizes = [5]
    hidden_dims = [64,128,256]
    dropouts = [0.3,0.4,0.5,0.6]
    data = load_data_and_make_label_sentence('results_payments_status_train.csv')
    game_features = joblib.load('embaddings/10.1train_for_bert_with_expert_behavioral_features.pkl')  # embaddings/new_maya_emb_for_val_new_features_{type}.pkl')#_with_bert#('1-10_bert_and_manual_avg_history_embadding.pkl')
    game_features = game_features.drop(['labels'], axis=1)
    game_features['labels'] = game_features['labels_for_probability']
    del game_features['labels_for_probability']
    all_data = game_features.merge(data, on=['pair_id'], how='left')
    del all_data['labels_y']
    del all_data['labels_for_probability']
    all_data['labels'] = all_data['labels_x']
    del all_data['labels_x']
    for batch in bath_sizes:
        data = cross_vali(data)
        for drop in dropouts:
            for hid_dim in hidden_dims:
                for fold in range(1,6):
                    training(fold,all_data, batch, hid_dim, drop)


