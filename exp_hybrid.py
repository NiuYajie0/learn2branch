
#%%
from sympy import false
import l2b.S01_generate_instances as S01_generate_instances
from l2b.hybrid_l2b import S02_generate_dataset, S03_train_gcnn_torch, S03_train_hybrid
from types import SimpleNamespace
import os


#%%
if __name__ == '__main__':

    os.environ['DGLBACKEND'] = 'pytorch'

    # %%

    problem = "setcover" # choices=['setcover', 'cauctions', 'facilities', 'indset']
    
    sampling_seed = 1
    
    train_seeds = range(6,7)
    gpu = 0 # CUDA GPU id (-1 for CPU).

    trainingSetSize = 'small' # choice=['small', 'large']

    ml_model_baseline = 'baseline_torch'

    ml_model_hybrid = 'film'

    # %%
    
    # S01_args = {
    #     'problem' : problem,
    #     'n_instances' : {'small': "(100, 20, 20, 20)",
    #                      'large': "(10000, 2000, 100, 2000)"}[trainingSetSize],
    #     'seed' : 0,
    #     'trainingSetSize' : trainingSetSize
    # }
    # S01_args = SimpleNamespace(**S01_args)
    # S01_generate_instances.exp_main(S01_args)


    # # %%
    # # 02 - Collect training samples
    # S02_args = {
    #     'problem' : problem,
    #     'seed' : sampling_seed,
    #     'njobs' : 7,
    #     'n_samples' : {'small':"(1000, 200, 200)",
    #                    'large_Gasse2019':"(100000, 20000, 20000)",
    #                    'large_Gupta2020':"(150000, 30000, 30000)"}[trainingSetSize], # Number of generated n_samples as (train_size, valid_size, test_size).
    #     # #             "(1000, 200, 200)"
    #     'trainingSetSize' : trainingSetSize
    # }
    # S02_args = SimpleNamespace(**S02_args)
    # S02_generate_dataset.exp_main(S02_args)

    # %%
    ## 03 - Train GCNN
    # for train_seed in train_seeds:
    #     S03_args = {
    #         'model' : ml_model_baseline,
    #         'gpu' : gpu,
    #         'problem' : problem,
    #         "l2": 0.0,
    #         # "data_path": "",
    #         # 'sampling' : samplingStrategy,
    #         'sample_seed' : sampling_seed,
    #         'seed' : train_seed, # python expression as string, to be used with eval(...)
    #         'trainingSetSize' : trainingSetSize
    #     }
    #     S03_args = SimpleNamespace(**S03_args)
    #     S03_train_gcnn_torch.exp_main(S03_args)

    ## 03 - Train GCNN
    for train_seed in train_seeds:
        S03_args = {
            'problem' : problem,
            'model' : ml_model_hybrid,
            'seed' : train_seed, # python expression as string, to be used with eval(...)
            'gpu' : gpu,
            'no_e2e': false,
            'distilled': false,
            'at': '', # choices: ['ED', 'MHE', '']
            'beta_at': 0,
            "l2": 0.0,
            # "data_path": "",
            # 'sampling' : samplingStrategy,
            'sample_seed' : sampling_seed,
            'trainingSetSize' : trainingSetSize
        }
        S03_args = SimpleNamespace(**S03_args)
        S03_train_hybrid.exp_main(S03_args)

    # # %%
    # ### 04 - Test branching accuracies w.r.t. strong branching
    # S04_args = {
    #     'gpu': gpu,
    #     'problem': problem,
    #     'sampling' : samplingStrategy,
    #     'sample_seed' : sampling_seed,
    #     'seeds' : train_seeds, # python expression as string, to be used with eval(...)
    # }
    # S04_args = SimpleNamespace(**S04_args)
    # S04_test.exp_main(S04_args)