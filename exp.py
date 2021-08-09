
#%%
import S01_generate_instances, S02_generate_dataset, S03_train_gcnn, S04_test
from types import SimpleNamespace

#%%
if __name__ == '__main__':

    # %%

    problem = "setcover" # choices=['setcover', 'cauctions', 'facilities', 'indset']
    # samplingStrategy = "depthK2" 
    
    train_seeds = "range(0,20)"
    gpu = 0 # CUDA GPU id (-1 for CPU).

    # %%
    ##  01 - Generate instances
    # S01_args = {
    #     'problem' : problem,
    #     'n_instances' : "(100, 20, 20, 20)",
    #     'seed' : 0,
    # }
    # S01_args = SimpleNamespace(**S01_args)
    # S01_generate_instances.exp_main(S01_args)

    
    samplingStrategies = ['depthK_adaptive2'] # # choices: uniform5, depthK, depthK2, depthK_adaptive
    sampling_seed = 0
    for samplingStrategy in samplingStrategies:

        # %%
        # 02 - Collect training samples
        S02_args = {
            'problem' : problem,
            'sampling' : samplingStrategy,
            'seed' : sampling_seed,
            'njobs' : 7,
            'n_samples' : "(1000, 200, 200)" # Number of generated n_samples as (train_size, valid_size, test_size).
            #             "(1000, 200, 200)"
        }
        S02_args = SimpleNamespace(**S02_args)
        S02_generate_dataset.exp_main(S02_args)

        # %%
        ## 03 - Train GCNN
        S03_args = {
            'model' : 'baseline',
            'gpu' : gpu,
            'problem' : problem,
            'sampling' : samplingStrategy,
            'sample_seed' : sampling_seed,
            'seeds' : train_seeds # python expression as string, to be used with eval(...)
        }
        S03_args = SimpleNamespace(**S03_args)
        S03_train_gcnn.exp_main(S03_args)

        # # %%
        # ### 04 - Test branching accuracies w.r.t. strong branching
        S04_args = {
            'gpu': gpu,
            'problem': problem,
            'sampling' : samplingStrategy,
            'sample_seed' : sampling_seed,
            'seeds' : train_seeds, # python expression as string, to be used with eval(...)
        }
        S04_args = SimpleNamespace(**S04_args)
        S04_test.exp_main(S04_args)