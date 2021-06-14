
#%%
import S01_generate_instances, S02_generate_dataset, S03_train_gcnn, S04_test

#%%
problem = "indset"
samplingStrategy = "uniform_5" # choices: uniform_5, depthK

#%%
S01_args = {}
S01_args['problem'] = problem
S01_args['n_instances'] = "(100, 20, 10, 20)" # python expression as string, to be used with eval(...) # Number of generated instances as (n_train_instances, n_valid_instances, n_transfer_instances, n_test_instances).
S01_generate_instances.exp_main(S01_args)


# strategies = ["uniform_5", "depthK"]
# for samplingStrategy in strategies:



#%%
S02_args = {}
S02_args['problem'] = problem
S02_args['sampling'] = samplingStrategy
S02_args['seed'] = 0
S02_args['njobs'] = 9
S02_args['n_samples'] = "(1000, 200, 200)" # Number of generated n_samples as (train_size, valid_size, test_size).
S02_generate_dataset.exp_main(S02_args)

# %%
S03_args = {}
S03_args['problem'] = problem
S03_args['sampling'] = samplingStrategy
S03_args['seeds'] = "range(0,10)" # python expression as string, to be used with eval(...)
S03_train_gcnn.exp_main(S03_args)

# %%
S04_args = {}
S04_args['problem'] = problem
S04_args['sampling'] = samplingStrategy
S04_args['seeds'] = "range(0,10)" # python expression as string, to be used with eval(...)
S04_test.exp_main(S04_args)