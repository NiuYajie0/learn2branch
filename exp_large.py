
#%%
import S01_generate_instances, S02_generate_dataset, S03_train_gcnn, S04_test
from types import SimpleNamespace

#%%
if __name__ == '__main__':

    # %%

    problem = "indset"
    samplingStrategy = "depthK" # choices: uniform5, depthK
    sample_seed = 1
    train_seeds = "range(0,10)"
    gpu = 0 # CUDA GPU id (-1 for CPU).

    #%%
    # S01_args = SimpleNamespace()
    # S01_args.problem = problem
    # S01_args.seed = 0
    # S01_args.n_instances = "(1000, 200, 100, 200)" # python expression as string, to be used with eval(...) # Number of generated instances as (n_train_instances, n_valid_instances, n_transfer_instances, n_test_instances).
    # S01_generate_instances.exp_main(S01_args)


    # # strategies = ["uniform5", "depthK"]
    # # for samplingStrategy in strategies:

    # #%%
    S02_args = SimpleNamespace()
    S02_args.problem = problem
    S02_args.sampling = samplingStrategy
    S02_args.seed = sample_seed
    S02_args.njobs = 9
    S02_args.n_samples = "(10000, 2000, 2000)" # Number of generated n_samples as (train_size, valid_size, test_size).
    S02_generate_dataset.exp_main(S02_args)

    # %%
    S03_args = SimpleNamespace()
    S03_args.model = 'baseline'
    S03_args.gpu = gpu
    S03_args.problem = problem
    S03_args.sampling = samplingStrategy
    S03_args.sample_seed = sample_seed
    S03_args.seeds = train_seeds # python expression as string, to be used with eval(...)
    S03_train_gcnn.exp_main(S03_args)

    # %%
    S04_args = {
        'gpu': gpu,
        'problem': problem,
        'sampling' : samplingStrategy,
        'sample_seed' : sample_seed,
        'seeds' : train_seeds, # python expression as string, to be used with eval(...)
    }
    S04_args = SimpleNamespace(**S04_args)
    # S04_args.problem = problem
    # S04_args.sampling = samplingStrategy
    # S04_args.sample_seed = sample_seed
    # S04_args.seeds = train_seeds 
    S04_test.exp_main(S04_args)