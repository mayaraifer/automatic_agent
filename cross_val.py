import random
def cross_vali(df):
    random.seed(10)
    ids = list(df['pair_id'].unique())
    random.shuffle(ids)
    ids1, ids2, ids3, ids4, ids5 = ids[:80], ids[80:160], ids[160:240], ids[240:320], ids[320:]
    df['fold'] = 0
    for index, row in df.iterrows():
        if row['pair_id'] in ids1:
            df.at[index, 'fold'] = 1
        elif row['pair_id'] in ids2:
            df.at[index, 'fold'] = 2
        elif row['pair_id'] in ids3:
            df.at[index, 'fold'] = 3
        elif row['pair_id'] in ids4:
            df.at[index, 'fold'] = 4
        else:
            df.at[index, 'fold'] = 5
    return df