# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import pandas as pd
import argparse

import glob
from scipy.stats import ttest_ind

# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
        default="facilities"
    )
    parser.add_argument(
        '--samplingStrategies',
        help='List of sampling strategies by python representation',
        choices=['uniform_5', 'depthK'],
        default="['uniform_5', 'depthK']"
    )

    args = parser.parse_args()

    # %%
    resultDir = 'results'
    problem = args.problem
    targets = eval(args.samplingStrategies)
    metric_columns = ['acc@1','acc@3','acc@5','acc@10']


    # %%
    samplingStragety1 = targets[0]
    samplingStragety2 = targets[1]

    targetfiles_strategy1 = glob.glob(f'{resultDir}/{problem}_{samplingStragety1}_*')
    targetfile1 = targetfiles_strategy1[-1]
    target_df1 = pd.read_csv(targetfile1)
    mean1 = target_df1[metric_columns].mean()
    std1 = target_df1[metric_columns].std()

    targetfiles_strategy2 = glob.glob(f'{resultDir}/{problem}_{samplingStragety2}_*')
    targetfile2 = targetfiles_strategy2[-1]
    target_df2 = pd.read_csv(targetfile2)
    mean2 = target_df2[metric_columns].mean()
    std2 = target_df2[metric_columns].std()

    t_statistics, p_values = ttest_ind(target_df1[metric_columns], target_df2[metric_columns], equal_var=False)


    # %%
    df = pd.DataFrame()
    df['Problem'] = [problem]*4
    df['Accuracy level'] = ['acc@1', 'acc@3', 'acc@5', 'acc@10']
    df[samplingStragety1] = ["%5.4f ± %5.4f" % (m*100, s*100) for (m, s) in zip(mean1, std1)]
    df[samplingStragety2] = ["%5.4f ± %5.4f" % (m*100, s*100) for (m, s) in zip(mean2, std2)]
    df['T-Test t-statistic'] = ["%5.4f" % p for p in t_statistics]
    df['T-Test p-value'] = ["%5.4f" % p for p in p_values]

    # %%
    df.to_csv(f'{resultDir}/{problem}_TTEST.csv')


    # %%



