from __future__ import division
from copy import deepcopy
import numpy as np
import torch
from update_and_push import update_inretaction,push_combi_to_interaction,take_an_action_with_svm_bert,take_an_action,take_an_action_with_bert,take_an_action_with_svm
from dm_model_with_bert import bert_eval_mcts
from create_bert_embadding import load_reviews_data_mcts
from dm_model_with_bert import BertTagger_mcts
import pandas as pd
import time
import joblib
from new_emb_try import create_embadding_for_qdn
from dm_models import LSTMTagger as LSTMDM
from dm_models import crf_eval as evalu
from Q_model import LSTMTagger as LSTMQ
from scipy.special import softmax
import random
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
manual_features_file =joblib.load('manual_binary_features_test_data.pkl')
bert_features_file = None
from sklearn.metrics import pairwise
from update_and_push import find_review_index_given_review


class NaughtsAndCrossesState():
    def __init__(self, interaction, interaction_id, model,currentRound,exp_tot_payoff,treshold,with_bert=None):
        self.board = interaction
        self.currentPlayer = 1
        self.last_rev_ind = 0
        self.initial_round = currentRound
        self.currentRound = currentRound
        self.interaction_id = interaction_id
        self.treshold =treshold
        self.model = model
        self.exp_tot_payoff = exp_tot_payoff
        self.isTerminal_ = False
        self.with_bert=with_bert
    def getCurrentPlayer(self):
        return self.currentPlayer
    def getPossibleActions(self):
        return[6,5,4,3,2,1,0]
    def getfeaturesfile(self):
        return manual_features_file, bert_features_file
    def takeAction(self, action,bert_model):
        newState = deepcopy(self)
        newState.board = push_combi_to_interaction(interaction=self.board,rounds = self.currentRound,action = action,interaction_id=self.interaction_id,this_round_payoff=0)
        newState.exp_tot_payoff = self.exp_tot_payoff
        newState.currentRound = self.currentRound
        newState.last_rev_ind = action
        if self.with_bert==False:
            interaction_embaded = create_embadding_for_qdn( newState.board, manual_features_file,True)
            prediction = evalu(interaction_embaded[[f'features_round_{i}' for i in range(1,newState.currentRound +1)]],self.model)[0][-1].item()
        else:
            interaction_embaded = create_embadding_for_qdn(newState.board, manual_features_file, False)
            columns = [f'features_round_{i}' for i in range(1, newState.currentRound  + 1)] + ['pair_id']
            interaction_embaded = interaction_embaded[columns]
            reveiws_data = load_reviews_data_mcts(newState.board[newState.board.index <  newState.currentRound ],  newState.currentRound )
            all_data = interaction_embaded.merge(reveiws_data, on=['pair_id'], how='left')
            prediction = bert_eval_mcts(all_data, bert_model)
        receiver_action = int(1 if prediction > self.treshold else 0)
        newState.exp_tot_payoff += receiver_action
        newState.board = update_inretaction(interaction = self.board,action = int(newState.last_rev_ind), round_ = newState.currentRound,this_round_payoff=receiver_action,interaction_id=self.interaction_id)
        newState.model = self.model
        newState.interaction_id =self.interaction_id
        newState.isTerminal_ = self.isTerminal_
        if newState.currentRound==10:
            newState.isTerminal_ = True
        else:
            newState.currentRound+=1
        return newState
    def isTerminal(self):
        if  self.currentRound == 10 and self.isTerminal_==True:
            return True
        else:
            return False
    def getReward(self):
        return self.exp_tot_payoff


def monte_carlo_tree_search( model_name,q_model_name, pairs):
    new_df_for_all_mcts = pd.DataFrame(
        columns=['pair_id', 'total_exp_payoff','total_dm_payoff','actions'])
    index = len(new_df_for_all_mcts)
    model = LSTMDM(embedding_dim=59, hidden_dim=256, tagset_size=1, dropout=0.3)
    value_model = LSTMQ(embedding_dim=59, hidden_dim=128, tagset_size=1, dropout=0.4)
    model.load_state_dict((torch.load(model_name )))
    model.eval()
    value_model.load_state_dict(torch.load(q_model_name))
    value_model = value_model.to(device)
    value_model.eval()
    bert_model = BertTagger_mcts(hidden_dim=64, tagset_size=1, dropout=0.5)
    bert_model.load_state_dict((torch.load('bert_models2/0.808_dm_bert_model_20_features_seed_3_hid_dim_64_drop_0.5_epoch_12.th')))
    bert_model = bert_model.to(device)
    bert_model.eval()
    tot_pay, count,receiver_payoff_tot = 0,0,0
    treshold=0.5
    sign,p='-',0
    timeLimit = 90000
    with_bert_in_mcts = False
    aginst_bert_dm_after=False
    explorationConstant=0.5
    for i in range(1,10):
        for pair_id in pairs:
            interaction = all_interaction[all_interaction['pair_id'] == pair_id]
            interaction = interaction.reset_index()
            exp_tot_payoff,receiver_payoff_tot = 0,0
            actions,probs = [], []
            for i in range(1, 11):
                start_time = time.time()
                initialState = NaughtsAndCrossesState(interaction_id=pair_id, model=model, interaction=interaction,
                                                      currentRound=i, exp_tot_payoff=exp_tot_payoff,treshold=treshold,
                                                      with_bert=with_bert_in_mcts)
                mcts1 = mcts(timeLimit=timeLimit,explorationConstant=explorationConstant)
                action = mcts1.search(initialState=initialState,value_model=value_model,
                                     manual_features_file=manual_features_file,bert_features_file=bert_features_file,bert_model=bert_model)
                print("--- %s seconds ---" % (time.time() - start_time))
                if aginst_bert_dm_after==False:
                    payoff, interaction, prob = take_an_action(action, i, interaction, pair_id,
                                                           model,manual_features_file, bert_features_file,p,sign)  # models[random.randrange(len(models))])
                else:
                    payoff, interaction, prob = take_an_action_with_bert(action, i, interaction, pair_id,bert_model,
                                                                         manual_features_file, bert_features_file,p,sign)
                receiver_payoff = interaction.at[i-1,'group_receiver_payoff']
                print(payoff,prob, action)
                exp_tot_payoff += payoff
                receiver_payoff_tot+=receiver_payoff
                probs.append(payoff)
                actions.append(action)
            new_df_for_all_mcts.loc[index] = [pair_id, exp_tot_payoff,receiver_payoff_tot, actions]
            index += 1
            tot_pay+=exp_tot_payoff
            count+=1


if __name__ == '__main__':
    all_interaction = pd.read_csv('results_payments_status_test.csv')
    all_interaction = all_interaction.loc[(all_interaction.status == 'play') & (all_interaction.player_id_in_group == 2)]
    all_interaction = all_interaction.drop_duplicates()
    pairs = list(all_interaction['pair_id'].unique())
    dm_model_name = 'dm_model_without_bert/0.824_20_behavioral_features+text_all_data_epoch_100_batch_5_hid_dim_256_drop_0.3.th'  # all_data_sigmoid_new_with_manual_updated_epoch_62_batch_20_hid_dim_100.th'#transformer_fscore_0.6552655759833379_acc_0.8458333333333333_epoch_98_batch_5_hid_dim_100_fold_'
    q_model_name = 'q_model_without_bert/0.389_20_features_behavioral+textual_all_data_epoch_100_batch_10_hid_dim_128_drop_0.4.th'  # all_data_sigmoid_new_with_manual_updated_epoch_62_batch_20_hid_dim_100.th'#transformer_fscore_0.6552655759833379_acc_0.8458333333333333_epoch_98_batch_5_hid_dim_100_fold_'
    monte_carlo_tree_search(model_name=dm_model_name,q_model_name=q_model_name, pairs=pairs)

