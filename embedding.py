import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np
import random
import scipy.sparse as sparse
import joblib
import sklearn.preprocessing as preprocessing
import copy
import time
#from language_prediction.train_test_models import *
import itertools
from collections import defaultdict
#from usful_scripts import *
def set_all_history_average_measures(data):
        """
        This function calculates some measures about all the history per round for each pair
        :return:
        """

        #print('Start set_all_history_average_measures')
        columns_to_calc = ['group_lottery_result', 'group_sender_payoff', 'lottery_result_high', 'chose_lose',
                           'chose_earn', 'not_chose_lose', 'not_chose_earn', '10_result']
        rename_columns = ['lottery_result', 'decisions', 'lottery_result_high', 'chose_lose', 'chose_earn',
                          'not_chose_lose', 'not_chose_earn', '10_result']
        # Create only for the experts and then assign to both players
        columns_to_chose = columns_to_calc + ['pair_id', 'subsession_round_number']
        data_to_create = data[columns_to_chose]
        data_to_create.columns = rename_columns + ['pair_id', 'subsession_round_number']
        ###add empty columns#####
        data_to_create = data_to_create.assign(history_decisions=None)
        data_to_create = data_to_create.assign(history_lottery_result_high=None)
        data_to_create = data_to_create.assign(history_lottery_result=None)
        data_to_create = data_to_create.assign(history_chose_lose=None)
        data_to_create = data_to_create.assign(history_chose_earn=None)
        data_to_create = data_to_create.assign(history_not_chose_lose=None)
        data_to_create = data_to_create.assign(history_not_chose_earn=None)
        data_to_create = data_to_create.assign(history_10_result=None)
        pairs = data_to_create.pair_id.unique()
        for pair in pairs:
            pair_data = data_to_create.loc[data_to_create.pair_id == pair]
            for round_num in range(2, 11):
                history = pair_data.loc[pair_data.subsession_round_number < round_num]
                weights = pow(alpha_global, round_num - history.subsession_round_number)
                for column in rename_columns:
                    if column == 'lottery_result':
                        j = 1
                    else:
                        j = 1
                    if alpha_global == 0:  # if alpha == 0: use average
                        data_to_create.loc[(data_to_create.pair_id == pair) &
                                           (data_to_create.subsession_round_number == round_num),
                                           f'history_{column}'] = round(history[column].mean(), 2)
                    else:
                        data_to_create.loc[(data_to_create.pair_id == pair) &
                                           (data_to_create.subsession_round_number == round_num),
                                           f'history_{column}'] = (pow(history[column], j) * weights).mean()
                    # for the first round put -1 for the history
                    data_to_create.loc[(data_to_create.pair_id == pair) &
                                       (data_to_create.subsession_round_number == 1), f'history_{column}'] = -1
        new_columns = [f'history_{column}' for column in rename_columns] + ['pair_id', 'subsession_round_number']
        data = data.merge(data_to_create[new_columns],  how='left')
        #print('Finish set_all_history_average_measures')

        return data

def create_all_columns(data,manual_features_file,use_text_features):
    # metri = np.array(list(manual_features_file['review_features'].values))
    # row_sums = metri.sum(axis=1)
    # new_matrix =  metri / row_sums[:, np.newaxis]
    #manual_features_file['review_features'] = list(new_matrix)
    data = data.loc[(data.status == 'play') & (data.player_id_in_group == 2)]
    # print(f'Number of rows in data: {self.data.shape[0]} after keep only play and decision makers')
    data = data.drop_duplicates()
    if use_text_features == True:
        for text_features in [manual_features_file]:
            if 'review' in list(text_features.columns):
                del text_features['review']
    data['exp_payoff'] = data.group_receiver_choice.map({1: 0, 0: 1})
    total_exp_payoff = data.groupby(by='pair_id').agg(
        total_exp_payoff=pd.NamedAgg(column='exp_payoff', aggfunc=sum))
    data = data.merge(total_exp_payoff, how='left', right_index=True, left_on='pair_id')
    data['10_result'] = np.where(data.group_lottery_result == 10, 1, 0)
    # data['4.17_hotel'] = np.where(data.group_average_score == 4.17, 1, 0)
    # data['6.66_hotel'] = np.where(data.group_average_score == 6.66, 1, 0)
    data['avg_hotel>8'] = np.where(data.group_average_score > 8, 1, 0)
    data['avg_hotel>8.5'] = np.where(data.group_average_score > 8.5, 1, 0)
    data['avg_hotel>9'] = np.where(data.group_average_score > 9, 1, 0)
    data['revealed_score>8'] = np.where(data.group_sender_answer_scores > 8, 1, 0)
    data['revealed_score>8.5'] = np.where(data.group_sender_answer_scores > 8.5, 1, 0)
    data['revealed_score>9'] = np.where(data.group_sender_answer_scores > 9, 1, 0)
    data['revealed_index>5'] = np.where(data.group_sender_answer_index > 5, 1, 0)
    data['revealed_index<3'] = np.where(data.group_sender_answer_index < 3, 1, 0)
    if use_text_features==True:
        manual_features_file.columns = ['review_id', 'manual_review_features']
        text_features=manual_features_file
    data = data[['revealed_score>9','revealed_score>8.5','revealed_score>8','revealed_index<3','revealed_index>5','avg_hotel>8','avg_hotel>9','avg_hotel>8.5','pair_id', 'total_exp_payoff', 'subsession_round_number', 'group_sender_answer_reviews',
               'exp_payoff', 'group_lottery_result', 'review_id','group_sender_answer_index',
               'group_average_score','lottery_result_low', 'lottery_result_med1','group_receiver_payoff','group_sender_payoff', 'lottery_result_high',
               'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn','previous_score', 'group_sender_answer_scores', '10_result']]#,'6.66_hotel','4.17_hotel']]
    #data = set_all_history_average_measures(data)
    if use_text_features==True:
        return data,text_features
    else:
        return data, None

def make_avg_history(data_pair):
    data_pair['avg_history_text'] = None
    for i in range(1,11):
        filtered_data =  data_pair[(data_pair.subsession_round_number<i) &(data_pair.subsession_round_number>=i-3)& (data_pair.group_sender_payoff==1)]
        if len(filtered_data['manual_review_features'].values)==0:
            data_pair.at[i-1,'avg_history_text']= np.array([0]*39)
        else:
            data_pair.at[i-1,'avg_history_text'] = np.average(np.array(list(filtered_data['manual_review_features'].values)), axis=0)
    return data_pair
def create_extra_features(data,features_file, use_manual):
    data = data.fillna(0)
    final_data = pd.DataFrame()
    #(len(data['pair_id'].unique()))
    for pair in data['pair_id'].unique():
        data_pair = data[data['pair_id']==pair]
        if len(data_pair)!=10:
            None
            #continue

        if str(features_file)!='None':
            data_pair=data_pair.merge(features_file, on =['review_id'],how = 'left')
        data_pair = data_pair.reset_index(drop=True)
        #data_pair['cum_lottery_result'] = data_pair['group_lottery_result'].cumsum()-data_pair['group_lottery_result']#/(data_pair['subsession_round_number']-1)
        #######data_pair['cum(avg)_reverse_lottery_result'] = (data_pair.loc[::-1, 'group_lottery_result'].cumsum()[::-1]-data_pair['group_lottery_result'])/(10-data_pair['subsession_round_number'])
        ############data_pair['cum(avg)_reverse_hotel_average_score'] = (data_pair.loc[::-1, 'group_average_score'].cumsum()[::-1]-data_pair['group_average_score'])/(10-data_pair['subsession_round_number'])
        #data_pair['cum(avg)_hotel_average_score'] = data_pair['group_average_score'].cumsum()/data_pair['subsession_round_number']
        #data_pair['cum(avg)_group_sender_answer_scores'] = data_pair['group_sender_answer_scores'].cumsum()/data_pair['subsession_round_number']
        data_pair['new_round_num'] = data_pair['subsession_round_number']/10
        data_pair['new_cum_lottery_result_low'] = (data_pair['lottery_result_low'].cumsum()-data_pair['lottery_result_low'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_lottery_result_med1'] = (data_pair['lottery_result_med1'].cumsum()-data_pair['lottery_result_med1'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_lottery_result_high'] = (data_pair['lottery_result_high'].cumsum()-data_pair['lottery_result_high'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_group_sender_payoff'] = (data_pair['group_sender_payoff'].cumsum() - data_pair['group_sender_payoff'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_group_receiver_payoff'] = (data_pair['group_receiver_payoff'].cumsum() - data_pair['group_receiver_payoff'])/(data_pair['subsession_round_number']-1)
        ##########
        data_pair['new_cum_chose_lose'] = (data_pair['chose_lose'].cumsum()-data_pair['chose_lose'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_chose_earn'] = (data_pair['chose_earn'].cumsum()-data_pair['chose_earn'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_not_chose_lose'] = (data_pair['not_chose_lose'].cumsum()-data_pair['not_chose_lose'])/(data_pair['subsession_round_number']-1)
        data_pair['new_cum_not_chose_earn'] = (data_pair['not_chose_earn'].cumsum()-data_pair['not_chose_earn'])/(data_pair['subsession_round_number']-1)
        data_pair['doesnt_chose_higther_9.5'] = data_pair.apply(
            lambda row: 1 if row['group_sender_answer_scores'] >= 9.5 and row['group_sender_payoff'] == 0 else 0,
            axis=1)
        data_pair['chose_less_than_7.5'] = data_pair.apply(
            lambda row: 1 if row['group_sender_answer_scores'] < 7.5 and row['group_sender_payoff'] == 1 else 0, axis=1)
        data_pair['new_cum_chose_less_than_7.5'] = (data_pair['chose_less_than_7.5'].cumsum() - data_pair[
            'chose_less_than_7.5']) / (data_pair['subsession_round_number'] - 1)  #
        data_pair['new_cum_doesnt_chose_higther_9.5'] = (data_pair['doesnt_chose_higther_9.5'].cumsum() - data_pair[
            'doesnt_chose_higther_9.5']) / (data_pair['subsession_round_number'] - 1)
        #################
        data_pair['new_group_average_score-8.5+'] = data_pair.apply(
            lambda x: 1 if x['group_average_score'] >= 8.5 else 0, axis=1)
        data_pair['new_group_average_score-7.5-8.5'] = data_pair.apply(
            lambda x: 1 if x['group_average_score'] < 8.5 and x['group_average_score'] >= 7.5 else 0, axis=1)
        data_pair['new_group_average_score-7.5-0'] = data_pair.apply(
            lambda x: 1 if x['group_average_score'] < 7.5 else 0, axis=1)

        #data_pair['cum_6.66_hotel'] = data_pair['6.66_hotel'].cumsum() - data_pair['6.66_hotel']
        #data_pair['cum_4.17_hotel'] = data_pair['4.17_hotel'].cumsum() -data_pair['4.17_hotel']
        #data_pair['take_less5'] = data_pair.apply(lambda row: 1 if row['group_average_score']<=5 and row['group_sender_payoff']==1 else 0, axis =1)
        #data_pair['take_less7.5more5'] = data_pair.apply(lambda row: 1 if (row['group_average_score']>5 or row['group_average_score']<7.5) and row['group_sender_payoff']==1 else 0, axis =1)
        #data_pair['chose_lose_in_10'] = data_pair.apply(lambda row: 1 if row['group_sender_answer_scores']==10 and row['chose_lose']==1 else 0, axis =1)

        #data_pair['new_cum_chose_lose_in_10'] = (data_pair['chose_lose_in_10'].cumsum()-data_pair['chose_lose_in_10'])/(data_pair['subsession_round_number']-1)
        #data_pair['new_cum_less7.5more5_take_hotel'] = (data_pair['take_less7.5more5'].cumsum()-data_pair['take_less7.5more5'])/(data_pair['subsession_round_number']-1)
        #data_pair['new_cum_less_5_take_hotel'] = (data_pair['take_less5'].cumsum()-data_pair['take_less5'])/(data_pair['subsession_round_number']-1)
        #data_pair['history_lottery_result'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['group_lottery_result'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['new_history_decisions'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['group_sender_payoff'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_lottery_result_high'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['lottery_result_high'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_lottery_result_low'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['lottery_result_low'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_lottery_result_med1'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['lottery_result_med1'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_chose_lose'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['chose_lose'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_chose_earn'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['chose_earn'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_not_chose_lose'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['not_chose_lose'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_not_chose_earn'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['not_chose_earn'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['new_history_10_result'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['10_result'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_group_average_score'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['group_average_score'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_group_sender_answer_scores'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['group_sender_answer_scores'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        #data_pair['history_group_sender_answer_index'] = data_pair.apply(lambda row: data_pair[data_pair.index < row.name]['group_sender_answer_index'].ewm(com=0.5).mean().values[-1] if row.name > 0 else 0, axis=1)/(data_pair['subsession_round_number']-1)
        data_pair['new_group_sender_answer_scores>8.5'] = data_pair.apply(lambda x: 1 if x['group_sender_answer_scores']>=8.5 else 0, axis=1)
        data_pair['new_group_sender_answer_scores8.5-7.5'] = data_pair.apply(lambda x: 1 if x['group_sender_answer_scores']<8.5 and x['group_sender_answer_scores']>=7.5 else 0, axis=1)
        data_pair['new_group_sender_answer_scores7.5'] = data_pair.apply(lambda x: 1 if  x['group_sender_answer_scores']<7.5 else 0, axis=1)
        data_pair['new_group_sender_answer_index<=3'] = data_pair.apply(lambda x: 1 if  x['group_sender_answer_index']<=3 else 0, axis=1)
        data_pair['new_group_sender_answer_index>3'] = data_pair.apply(lambda x: 1 if  x['group_sender_answer_index']>3 else 0, axis=1)
        data_pair['next_exp_payoff'] = data_pair['total_exp_payoff'] - (data_pair['exp_payoff'].cumsum() - data_pair['exp_payoff'])

        #new = [col for col in data_pair.columns if 'new_' in col]

        new = ['new_round_num', 'new_cum_lottery_result_low', 'new_cum_lottery_result_med1', 'new_cum_lottery_result_high', 'new_cum_group_sender_payoff', 'new_cum_group_receiver_payoff', 'new_cum_chose_lose', 'new_cum_chose_earn', 'new_cum_not_chose_lose', 'new_cum_not_chose_earn', 'new_cum_chose_less_than_7.5', 'new_cum_doesnt_chose_higther_9.5', 'new_group_average_score-8.5+', 'new_group_average_score-7.5-8.5', 'new_group_average_score-7.5-0','new_group_sender_answer_scores>8.5','new_group_sender_answer_scores8.5-7.5','new_group_sender_answer_scores7.5','new_group_sender_answer_index<=3','new_group_sender_answer_index>3']#, 'new_group_sender_answer_scores>9.5', 'new_group_sender_answer_scores9.5-7.5', 'new_group_sender_answer_scores7.5', 'new_group_sender_answer_index<=3', 'new_group_sender_answer_index>3']
        #print(len(new))
        #print(new)
        data_pair = data_pair.fillna(0)
        data_pair = data_pair.replace(-np.inf, 0)
        data_pair = data_pair.replace(np.inf, 0)
        features = pd.DataFrame(columns=['total_cum_features'])
        if use_manual==True:
            features['total_cum_features'] = data_pair.apply(lambda row:[row[col] for col in new]+list(row['manual_review_features']),axis=1)#list(row['manual_review_features'])+list(row['avg_history_text'])# list(np.array([row[col] for col in new])/(np.array([row[col] for col in new]).sum()))
        else:
            features['total_cum_features'] = data_pair.apply(lambda row: [row[col] for col in new],axis=1)
       #print(len(features.at[0,'total_cum_features']))
        new_column_name = {}
        #print(features)
        for val in range(0, 10):
            new_column_name[val] = f"features_round_{val + 1}"
        features = features.T.rename(columns=new_column_name)
        features = features.reset_index(drop=True)
        features.at[0, 'pair_id'] = pair
        features['labels'] = None
        features['labels'] = features['labels'].astype(object)
        features.at[0,'labels'] = list(data_pair['next_exp_payoff'].values)
        features['labels_for_probability'] = None
        features['labels_for_probability'] = features['labels_for_probability'].astype(object)
        features.at[0, 'labels_for_probability'] = list(data_pair['exp_payoff'].values)
        final_data = pd.concat([final_data, features], axis=0, ignore_index=True)
    return final_data

def create_embadding_for_qdn(df,manual_features_file, use_manual):
    # metri = np.array(list(manual_features_file['review_features'].values))
    # scaler = preprocessing.StandardScaler().fit(metri)
    # new_matrix = scaler.transform(metri)
    # manual_features_file['review_features'] = list(new_matrix)
    # joblib.dump(manual_features_file,'manual_binary_norm_features_train_data.pkl')
    data, text_features = create_all_columns(df,manual_features_file ,use_manual)
    final_data = create_extra_features(data,text_features, use_manual)
    #joblib.dump(final_data,'embaddings/10.1train_with_expert_behavioral_features.pkl')
    #joblib.dump(final_data,'embaddings/12.5test_only_numeric.pkl')
    #print('the file is saved!')
    return final_data
if __name__ == '__main__':
    #bert_features_file = joblib.load('new_bert_embedding_test_data.pkl')
    manual_features_file = joblib.load('manual_binary_features_train_data.pkl')
    df = pd.read_csv('results_payments_status_num_test_data.csv')#('results_payments_status_train.csv')
        #df = pd.read_csv('results_payments_status_numeric.csv')
    #for col in df.columns:
    #    if col not in df1.columns:
    #        del df[col]
    #df.to_csv('results_payments_status_num_test_data.csv', index = False)
    #print(len(df))
    create_embadding_for_qdn(df,manual_features_file, False)#['history_features','cum_features','general_features'])
