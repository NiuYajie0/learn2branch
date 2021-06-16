import S01_generate_instances, S02_generate_dataset, S03_train_gcnn, S04_test

#%%
problem = "facilities"

S01_args = {}
S01_args.problem = problem
S01_args.n_instances = "(1000, 200, 100, 200)" # python expression as string, to be used with eval(...)
S01_generate_instances.exp_main(S01_args)


# strategies = ["uniform_5", "depthK"]
# for samplingStrategy in strategies:

#%%
samplingStrategy = "uniform_5" # choices: uniform_5, depthK

S02_args = {}
S02_args.problem = problem
S02_args.sampling = samplingStrategy
S02_args.seed = 0
S02_args.njobs = 9
S02_generate_dataset.exp_main(S02_args)

# %%
S03_args = {}
S03_args.problem = problem
S03_args.sampling = samplingStrategy
S03_args.seeds = "range(0,10)" # python expression as string, to be used with eval(...)
S03_train_gcnn.exp_main(S03_args)

# %%
S04_args = {}
S04_args.problem = problem
S04_args.sampling = samplingStrategy
S04_args.seeds = "range(0,10)" # python expression as string, to be used with eval(...)
S04_test.exp_main(S04_args)