import argparse
import configparser
import copy

import logging

import numpy as np
import os
import time
from pathlib import Path

os.environ['NUMEXPR_MAX_THREADS'] = r'12'
proc_num = 12
t_str = time.strftime("%Y%m%d%H%M%S")


def currentDir():
    return os.path.dirname(os.path.realpath(__file__))


def parentDir(mydir):
    return str(Path(mydir).parent.absolute())


def init_logging(logfile):
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()

    logger = logging.getLogger()
    logger.handlers = []

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ch)


def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_default = config['DEFAULT']
    parameters['data_path'] = parentDir(currentDir()) + os.sep + 'datasets' + os.sep + 'iris' + \
                              os.sep + p_default['DataFile']
    parameters['num_exps'] = p_default.getint('RunningTimes')

    # add time stamp to the name of log file
    logfile = p_default['LogFile']
    index = logfile.rfind('.')

    if index != -1:
        logfile = logfile[:index] + "_" + "unknown_" + t_str + logfile[index:]
    else:
        logfile = logfile + "_" + "unknown_" + t_str + ".log"

    parameters['logpath'] = currentDir() + os.sep + "log" + os.sep + logfile

    return parameters


def load_data(path):
    # Solve iris
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # train validation test: 0.6 0.2 0.2
    df = pd.read_csv(path, header=None)

    # Column names
    df.columns = ['SepalLength', 'SepalWidth',
                  'PetalLength', 'PetalWidth', 'Class']

    # Changes string to float
    df.SepalLength = df.SepalLength.astype(float)
    df.SepalWidth = df.SepalWidth.astype(float)
    df.PetalLength = df.PetalLength.astype(float)
    df.PetalWidth = df.PetalWidth.astype(float)

    # Sets label name as Y
    df = df.rename(columns={'Class': 'Y'})
    x_tv, x_test, y_tv, y_test = train_test_split(df.drop(columns=['Y']).values, df.Y.values, test_size=0.2,
                                                  random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.25,
                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)
    runs = parameters['num_exps']
    # init logging
    init_logging(parameters['logpath'])

    logging.critical('=================')
    logging.critical('dataset: %s', parameters['data_path'])
    logging.critical('=================')

    # load dataset
    x_tr, y_tr, x_val, y_val, x_test, y_test = load_data(parameters['data_path'])

    import sys

    sys.path.append('..')
    import sshap
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    iris_seed = 23  # + c_run # if seed random, move this part in the loop

    logging.critical(f'seed {iris_seed}')
    model = LogisticRegression(max_iter=100) # model = SVC(decision_function_shape='ovo')
    idxes = list(np.arange(len(y_tr)))
    sshap.reproduce(seed=iris_seed)
    np.random.shuffle(idxes)
    ls = sshap.ShardedStruct(depth=1, nl=[idxes[:len(y_tr) // 3], idxes[len(y_tr) // 3:len(y_tr) // 3 * 2],
                                        idxes[len(y_tr) // 3 * 2:]])
    # ls = sshap.ShardedStruct(depth=1, nl=[idxes[:len(y_tr) // 2], idxes[len(y_tr) // 2:]])

    perf_runs = []
    for c_run in range(runs):
        sshap.reproduce(seed=42 + c_run)
        logging.critical('start valuation, current runs %d / %d' % (c_run + 1, runs))
        # cal different value
        loo = sshap.loo(x_tr, y_tr, x_val, y_val, model, ls)
        logging.critical('loo complete')
        sv, beta41, beta161, beta14, beta116 = sshap.monte_carlo_sv_beta(x_tr, y_tr, x_val, y_val, model, ls,
                                                                         m=10 * len(y_tr), proc_num=proc_num)
        logging.critical('sv and beta sv complete')
        ssv = sshap.monte_carlo_ssv(x_tr, y_tr, x_val, y_val, model, ls, m=10 * len(y_tr), proc_num=proc_num)
        logging.critical('ssv complete')
        rand_val = np.random.random(size=len(y_tr))
        logging.critical('random complete')

        np.savez('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_iris_val.npz', loo=loo, rand_val=rand_val, sv=sv,
                 ssv=ssv, beta161=beta161, beta41=beta41,
                 beta14=beta14, beta116=beta116)
        np.savez('./result/' + t_str + f'_{str(c_run)}_{str(runs)}_' + 'iris_val.npz', loo=loo, rand_val=rand_val,
                 sv=sv, ssv=ssv, beta161=beta161, beta41=beta41,
                 beta14=beta14, beta116=beta116)

        # check the perf on test dataset
        perf_lists = []
        acc_sub_list = []
        vales = np.load('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_iris_val.npz')
        for j, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
            perf_lists.append([])
            l2h = np.argsort(vales[alg])
            for i in range(len(y_tr), 0, -1):
                tmp_ls = copy.deepcopy(ls)
                tmp_ls.idxes_available = l2h[:i]
                acc, _, _ = sshap.eval_utility(x_tr, y_tr, x_test, y_test, model, tmp_ls)
                perf_lists[j].append(acc)
            # Do subsample sample 30% in each shard
            # sampled_set_idx = set()
            # for s in ls.nl:
            #     val = vales[alg][s]  # value in this shard
            #     idx_in_s = np.argsort(val)[::-1][:int(len(s)*0.3)]
            #     sampled_s = (np.asarray(s))[idx_in_s]
            #     sampled_set_idx = sampled_set_idx | set(sampled_s)
            # assert len(sampled_set_idx) == int(len(y_tr)*0.3)
            # tmp_ls = copy.deepcopy(ls)
            # tmp_ls.idxes_available = list(sampled_set_idx)
            # acc_sub = sshap.eval_utility(x_tr, y_tr, x_test, y_test, model, tmp_ls)
            # acc_sub_list.append(acc_sub)
        np.savetxt('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_perf_list.txt', np.asarray(perf_lists))
        np.savetxt('./result/' + t_str + f'_{str(c_run)}_{str(runs)}_' + 'perf_list.txt', np.asarray(perf_lists))
        perf_runs.append(perf_lists)

    np.save('./result/' + 'latest_perf_runs.npy', np.asarray(perf_runs))
    np.save('./result/' + t_str + 'perf_runs.npy', np.asarray(perf_runs))
