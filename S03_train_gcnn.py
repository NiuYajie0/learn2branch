import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import tensorflow as tf
# import tensorflow.contrib.eager as tfe

import utilities
from utilities import log

from utilities_tf import load_batch_gcnn


def load_batch_tf(batch_filename_tensor):
    # 在这个
    return tf.py_function(
        load_batch_gcnn,
        [batch_filename_tensor],
        [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32])


def pretrain(model, dataloader):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : tf.data.Dataset
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        for batch in dataloader:
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            # model: GCNPolicy(BaseModel) --- pre_train 方法定义在 BaseModel 里面
            # pre_train 实际上就是跑 model.call, 但是过程中只要抛出了 PreNormLayerException 就接住并返回True
            # PreNormLayerException 是在某个 PreNormLayer 的call中更新后抛出的（必须处在 waiting_updates 中才能更新）
            # 所以在多个 PreNormLayer 处于 waiting_updates 状态的时候，调用这个方法只会有一个 PreNormLayer 被更新了
            #    （最前面的、还在waiting updates的PreNormLayer），后面的 layers 也都不会有计算
            # 一次循环是用一个 batch 更新最前面的 PreNormLayer 的参数，只要那个 PreNormLayer 还在 waiting_updates 状态，就还是更新那个 PreNormLayer
            # 所以这整个 for batch in dataloader 循环跑完就是用 pretraindata 数据集里面的所有数据更新最前面那个 PreNormLayer 的参数
            # 因此要重复几次这个循环，用 i 来记次数，共 7 个 PreNormLayer, 跑完 i 就等于 7
            # 那么 pre_train_next 方法要做的就是，把最前面的那个还在等待更新的 PreNormLayer 的等待状态关了，即把 waiting_updates 设为 False
            if not model.pre_train(batched_states, tf.convert_to_tensor(True)):
                break

        # 按顺序访问每一个 layers，对第一个处在 waiting_updates 和 received_updates 状态的 PreNormLayer 运行 layer.stop_updates()
        # 即把最前面那个 PreNormLayer 的等待状态关了（同时设好 self.shift 和 self.var）
        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer, name = res

        i += 1

    return i


def process(model, dataloader, top_k, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        # 这里的shapes中有一个维度都会比较大，因为这是concatenate以后的大图的信息
        # ev 对应 edge_features，不知道为什么用 ev 而不是 ef; 
        # cands是把一个batch中的每个sample的candidates都并起来得到的，cands.shape[0]就等于n_cands中所有元素的和
        # c:(55394,5), ei:(2, 363081), ev:(363081, 1), v:(80800, 19), n_cs:(8,), 
        # n_vs:(8,), n_cands:(8,), cands:(315,), best_cands:(8,), cand_scores:(315,)
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
        batched_states = (c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True))  # prevent padding
        batch_size = n_cs.shape[0]

        if optimizer:
            with tf.GradientTape() as tape:
                logits = model(batched_states, tf.convert_to_tensor(True)) # training mode # logits:(1,80800)对应变量的数目
                
                # squeeze(logits, 0)把大小为1的维度去掉，然后tf.gather(tf.squeeze(logits, 0), cands)从logits中按照cands获取cands的logits值, logits:(315,)；执行expand_dim后, logits:(1, 315)
                # 这里的 cands 是在 utilities_tf.load_batch_gcnn 中 shift 过了的
                logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables # tf.
                logits = model.pad_output(logits, n_cands.numpy())  # apply padding now # logits:(8,53)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=best_cands, logits=logits)
            grads = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            logits = model(batched_states, tf.convert_to_tensor(False))  # eval mode
            logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=best_cands, logits=logits)

        true_scores = model.pad_output(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)
        true_scores = true_scores.numpy()
        true_bestscore = true_bestscore.numpy()

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(logits, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += np.sum(loss) * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc

def exp_main(args):
    seeds = eval(args.seeds)
    
    ### NUMPY / TENSORFLOW SETUP ###
    ## TODO 下面这个没用了，要用tf.config.set_visible_devices来控制是否使用CPU
    ## 见 https://www.tensorflow.org/api_docs/python/tf/config/set_visible_devices
    if args.gpu == -1:
        tf.config.set_visible_devices(tf.config.list_physical_devices('CPU')[0])
    else:
        cpu_devices = tf.config.list_physical_devices('CPU') 
        gpu_devices = tf.config.list_physical_devices('GPU') 
        tf.config.set_visible_devices([cpu_devices[0], gpu_devices[args.gpu]])
        tf.config.experimental.set_memory_growth(gpu_devices[args.gpu], True)

    # 可能还需要设定最大虚拟GPU大小，用 tf.config.experimental.set_virtual_device_configuration
    # 见 https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth 


    for seed in seeds:

        ### HYPER PARAMETERS ###
        max_epochs = 300
        epoch_size = 20
        batch_size = 8 # 我的电脑设为16就内存溢出 -- 因此目前这个方法更适合规模小一些的问题 # TODO 用他们后面的那种改进
        pretrain_batch_size = 16
        valid_batch_size = 16
        lr = 0.001
        patience = 10
        early_stopping = 20
        top_k = [1, 3, 5, 10]
        train_ncands_limit = np.inf
        valid_ncands_limit = np.inf
        sampling_strategy = args.sampling
        

        problem_folders = {
            'setcover': f'setcover/500r_1000c_0.05d({sampling_strategy})/{args.sample_seed}',
            'cauctions': f'cauctions/100_500({sampling_strategy})/{args.sample_seed}',
            'facilities': f'facilities/100_100_5({sampling_strategy})/{args.sample_seed}', # TODO
            'indset': f'indset/500_4({sampling_strategy})/{args.sample_seed}',
        }
        problem_folder = problem_folders[args.problem]

        # running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"
        running_dir = f"trained_models/{args.problem}/{sampling_strategy}/ss{args.sample_seed}/ts{seed}" # TODO

        os.makedirs(running_dir)

        ### LOG ###
        logfile = os.path.join(running_dir, 'log.txt')

        log(f"max_epochs: {max_epochs}", logfile)
        log(f"epoch_size: {epoch_size}", logfile)
        log(f"sampling: {sampling_strategy}", logfile)
        log(f"batch_size: {batch_size}", logfile)
        log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
        log(f"valid_batch_size : {valid_batch_size }", logfile)
        log(f"lr: {lr}", logfile)
        log(f"patience : {patience }", logfile)
        log(f"early_stopping : {early_stopping }", logfile)
        log(f"top_k: {top_k}", logfile)
        log(f"problem: {args.problem}", logfile)
        log(f"gpu: {args.gpu}", logfile)
        log(f"seed {seed}", logfile)


        tf.config.run_functions_eagerly(True)

        rng = np.random.default_rng(seed)
        tf.random.set_seed(rng.integers(np.iinfo(int).max))

        ### SET-UP DATASET ###
        train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))


        def take_subset(sample_files, cands_limit):
            nsamples = 0
            ncands = 0
            for filename in sample_files:
                with gzip.open(filename, 'rb') as file:
                    sample = pickle.load(file)

                _, _, _, cands, _ = sample['data']
                ncands += len(cands)
                nsamples += 1

                if ncands >= cands_limit:
                    log(f"  dataset size limit reached ({cands_limit} candidate variables)", logfile)
                    break

            return sample_files[:nsamples]


        if train_ncands_limit < np.inf:
            train_files = take_subset(rng.permutation(train_files), train_ncands_limit)
        log(f"{len(train_files)} training samples", logfile)
        if valid_ncands_limit < np.inf:
            valid_files = take_subset(valid_files, valid_ncands_limit)
        log(f"{len(valid_files)} validation samples", logfile)

        train_files = [str(x) for x in train_files]
        valid_files = [str(x) for x in valid_files]

        valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
        valid_data = valid_data.batch(valid_batch_size)
        valid_data = valid_data.map(load_batch_tf)

        # TODO Debug
        # load_batch_tf([tf.constant('data\\samples\\facilities\\100_100_5\\train\\sample_1.pkl')])

        valid_data = valid_data.prefetch(1)

        pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
        pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
        pretrain_data = pretrain_data.batch(pretrain_batch_size)
        pretrain_data = pretrain_data.map(load_batch_tf)
        pretrain_data = pretrain_data.prefetch(1)

        ### MODEL LOADING ###
        model = importlib.import_module(f'models.{args.model}.model')
        # sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
        # import model
        # importlib.reload(model)
        model = model.GCNPolicy()
        # del sys.path[0]
        

        ### TRAINING LOOP ###
        optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: lr)  # dynamic LR trick
        best_loss = np.inf
        for epoch in range(max_epochs + 1):
            log(f"EPOCH {epoch}...", logfile)
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()

            # TRAIN
            if epoch == 0:
                n = pretrain(model=model, dataloader=pretrain_data)
                log(f"PRETRAINED {n} LAYERS", logfile)
                # model compilation
                # model.call = tf.function(model.call, input_signature=model.input_signature)
            else:
                # bugfix: tensorflow's shuffle() seems broken...
                epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
                train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
                train_data = train_data.batch(batch_size)
                train_data = train_data.map(load_batch_tf)
                train_data = train_data.prefetch(1)
                train_loss, train_kacc = process(model, train_data, top_k, optimizer)
                log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

            # TEST
            # 第一次 valid: VALID LOSS: 39.318  acc@1: 0.440 acc@3: 0.785 acc@5: 0.915 acc@10: 0.995
            # 所以 acc@10 本身就很高; 不过和论文中也匹配，facilities这类问题acc@10就是很高
            valid_loss, valid_kacc = process(model, valid_data, top_k, None)
            log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

            if valid_loss < best_loss:
                plateau_count = 0
                best_loss = valid_loss
                model.save_state(os.path.join(running_dir, 'best_params.pkl'))
                log(f"  best model so far", logfile)
            else:
                plateau_count += 1
                if plateau_count % early_stopping == 0:
                    log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                    break
                if plateau_count % patience == 0:
                    lr *= 0.2
                    log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

        model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='baseline',
    )
    parser.add_argument(
        '-s', '--seeds',
        help='Random generator seeds as a python list or range representation.',
        # type=utilities.valid_seed,
        default="range(0,5)",
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--sampling',
        help='Sampling Strategy',
        choices=['uniform5', 'depthK', 'depthK2'],
        default='uniform5'
    )
    parser.add_argument(
        '--sample_seed',
        help='seed of the sampled data',
        type=utilities.valid_seed,
        default=0
    )
    args = parser.parse_args()

    exp_main(args)

