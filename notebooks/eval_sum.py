# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import os
# os.chdir("../")


# %%
import glob
import pandas as pd
import time
import numpy as np
from scipy import stats


# %%
resultDir = 'results'
problem = 'setcover' # choices=['setcover', 'cauctions', 'facilities', 'indset']
sampling_Strategies = ['uniform5','depthK','depthK2'] # choices: uniform5, depthK, depthK2, depthK3
seeds = [0,1,2,3,4]


# %%
eval_files = glob.glob(f'{resultDir}/{problem}_*.csv')
eval_file = eval_files[-1]

df = pd.read_csv(eval_file)
df = pd.concat([df[df['type']=='small'], df[df['type']=='medium']])
df = df.astype({'nlps': float, 'nnodes' : float})

df_gcnns = df[df['policy'] != 'internal:relpscost']


# %%
def gmean_1shifted(x):
    return stats.mstats.gmean(x + 1) - 1

# %% [markdown]
# # 1. Means

# %%
dfgcnns_gmean = df_gcnns.groupby(['type','sampling_strategy'])[['nnodes', 'stime']].agg(gmean_1shifted)
dfgcnns_gmean


# %%
df_list = []
for probSize in dfgcnns_gmean.index.levels[0]:
    df_list.append(dfgcnns_gmean.loc[probSize] / dfgcnns_gmean.loc[(probSize, 'uniform5')])
dfgcnns_gmean_normalized = pd.concat(df_list, keys=dfgcnns_gmean.index.levels[0])
dfgcnns_gmean_normalized

# %% [markdown]
# # 2. Std variances (per instance)

# %%
dfgcnns_std_perInstance = df_gcnns.groupby(['type','sampling_strategy','instance']).std() / df_gcnns.groupby(['type','sampling_strategy','instance']).mean()


# %%
dfgcnns_std_mean = dfgcnns_std_perInstance.groupby(['type','sampling_strategy'])[['nnodes','stime']].mean()
dfgcnns_std_mean

# %% [markdown]
# # 3. 计算Wins

# %%
def get_winner_indices(x):
    # return min(x['stime'])
    return x.idxmin()


# %%
df_gcnns.groupby(['type','instance','seed'])['stime'].agg(pd.Series.idxmin)


# %%
df_uniform5 = df_gcnns[df_gcnns['sampling_strategy']=='uniform5']
df_depthK = df_gcnns[df_gcnns['sampling_strategy']=='depthK']
df_depthK2 = df_gcnns[df_gcnns['sampling_strategy']=='depthK2']
df_gcnns['Wins'] = 0


# %%
df_gcnns


# %%
for i in range(0,len(df_uniform5)):
    uniform5_row = df_uniform5.iloc[i]
    depthK_row = df_gcnns[
        (df_gcnns['sampling_strategy'] == 'depthK') &
        (df_gcnns['policy'] == uniform5_row.policy) &
        (df_gcnns['seed'] == uniform5_row.seed) &
        (df_gcnns['type'] == uniform5_row.type) &
        (df_gcnns['instance'] == uniform5_row.instance)
    ]
    depthK2_row = df.loc[
        (df_gcnns['sampling_strategy'] == 'depthK2') &
        (df_gcnns['policy'] == uniform5_row.policy) &
        (df_gcnns['seed'] == uniform5_row.seed) &
        (df_gcnns['type'] == uniform5_row.type) &
        (df_gcnns['instance'] == uniform5_row.instance)
    ]
    winner = np.argmin([uniform5_row.stime, depthK_row.stime.iloc[0], depthK2_row.stime.iloc[0]])
    winner = ['uniform5', 'depthK', 'depthK2'][winner]
    df_gcnns.loc[ (df_gcnns['sampling_strategy'] == winner) &
            (df_gcnns['policy'] == uniform5_row.policy) &
            (df_gcnns['seed'] == uniform5_row.seed) &
            (df_gcnns['type'] == uniform5_row.type) &
            (df_gcnns['instance'] == uniform5_row.instance), 'Wins'] += 1


# %%
df_gcnns.groupby(['type', 'sampling_strategy']).sum()


# %%



