import torch
import torch.nn as nn
import joblib
import random
import numpy as np
import copy
from sklearn import metrics
import time
from cross_val import cross_vali
epoch_num = 100
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

class LSTMHC(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size,dropout):
        super(LSTMHC, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence_emb):
        others, text = sentence_emb[:, :, :20], self.sigmoid(sentence_emb[:, :, 20:])
        sentence_emb = torch.cat([others, text], dim=2)
        lstm_out, _ = self.lstm(sentence_emb)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

def crete_random_batches(train_df, batch_size):
    training_data, batch, count = [], [], 0
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
    embedding_dim = len(data.loc[0]['features_round_1'])
    print(embedding_dim)
    model = LSTMHC(embedding_dim=embedding_dim, hidden_dim=hidden_dim, tagset_size=1,dropout=dropout)
    model = model.to(device)
    optimizer =  torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    training_data, testing_data, batch, batch1, pairs_train = [], [], [], [], []
    train_df = data[data['fold']!=fold]
    epoch_counter=0
    max_f_score_for_fold ,min_loss= 0,100
    for epoch in range(epoch_num):
        epoch_counter+=1
        training_data = crete_random_batches(train_df, batch_size)
        tot_loss=0
        for batch in training_data:
            sentence_ = [sample[0] for sample in batch]
            tags_ = [sample[1] for sample in batch]
            sentence = torch.Tensor(sentence_).to(device)
            model.zero_grad()
            tag_scores = model(sentence)
            real_teg = torch.Tensor(tags_).to(device)
            loss = loss_function(tag_scores.reshape(real_teg.shape[0],10), real_teg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            tot_loss+=loss.item()

def main():
    bath_sizes = [10, 15, 25, 20, 5]
    hidden_dims = [64, 128, 256]
    dropouts = [0.4, 0.3, 0.5, 0.6]
    type = '10.1train_with_expert_behavioral_features'
    data = joblib.load(f'embaddings/{type}.pkl')
    del data['labels']
    data['labels'] = data['labels_for_probability']
    del data['labels_for_probability']
    data = cross_vali(data)
    for batch in bath_sizes:
        for drop in dropouts:
            for hid_dim in hidden_dims:
                set_seed(111)
                for fold in range(1, 6):
                    training(fold, data, batch, hid_dim, drop)
