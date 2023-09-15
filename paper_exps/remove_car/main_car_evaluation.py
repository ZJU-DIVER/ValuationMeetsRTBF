import argparse
import configparser
import copy

import logging

import numpy as np
import os
import time
from pathlib import Path

runs = 3
proc_num = 10
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
    parameters['data_path'] = parentDir(currentDir()) + os.sep + 'datasets' + os.sep + 'car_evaluation' + os.sep + p_default['DataFile'] #1
    parameters['num_target_features'] = p_default.getint('NumOfFeaturesToRecover')
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


def load_data(path): #2
    # Solve iris
    import pandas as pd
    from sklearn.model_selection import train_test_split

    import category_encoders as ce
    # train validation test: 0.6 0.2 0.2
    # df = pd.read_csv(path, header=None)

    # # Column names
    # df.columns = ['SepalLength', 'SepalWidth',
    #               'PetalLength', 'PetalWidth', 'Class']

    # # Changes string to float
    # df.SepalLength = df.SepalLength.astype(float)
    # df.SepalWidth = df.SepalWidth.astype(float)
    # df.PetalLength = df.PetalLength.astype(float)
    # df.PetalWidth = df.PetalWidth.astype(float)

    # # Sets label name as Y
    # df = df.rename(columns={'Class': 'Y'})
    # x_tv, x_test, y_tv, y_test = train_test_split(df.drop(columns=['Y']).values, df.Y.values, test_size=0.2,
    #                                               random_state=42)
    # x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.25,
    #                                                   random_state=42)
    # return x_train, y_train, x_val, y_val, x_test, y_test

    df = pd.read_csv(path, header=None)

    df.replace('5more','5',inplace=True)

    # Column names
    df.columns = ['buying', 'maint',
                    'doors', 'persons', 'lug_boot', 'safety', 'Decision']

    # Changes string to float
    # df.SepalLength = df.SepalLength.astype(float)
    # df.SepalWidth = df.SepalWidth.astype(float)
    # df.PetalLength = df.PetalLength.astype(float)
    # df.PetalWidth = df.PetalWidth.astype(float)

    # Sets label name as Y
    df = df.rename(columns={'Decision': 'class'})
    def show(df):
        for i in df.columns[1:]:
            print("Feature: {} with {} Levels".format(i,df[i].unique()))

    show(df)
    X = df.drop('class', axis = 1)
    y = df['class']
    # print(X,y)
    # x_train_array, x_test_array, y_train, y_test = train_test_split(X, y,
    #                                                                 test_size=728,
    #                                                                 random_state=42)
    # print(x_train_array.values, x_test_array.values, y_train, y_test)

    x_tv, x_test, y_tv, y_test = train_test_split(X, y, test_size=0.2,
                                                  random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.25,
                                                      random_state=42)
    x_train, a, y_train, b = train_test_split(x_train, y_train, test_size=36,
                                                      random_state=42)                                                                   
    encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    x_train = encoder.fit_transform(x_train)
    x_test = encoder.transform(x_test)
    x_val = encoder.transform(x_val)
    return x_train.values, y_train.values, x_val.values, y_val.values, x_test.values, y_test.values


if __name__ == '__main__':
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)

    # init logging
    init_logging(parameters['logpath'])
    # logging.info("This should be only in file")
    # logging.critical("This shoud be in both file and console")

    logging.critical('=================')
    logging.critical('dataset: %s', parameters['data_path'])
    logging.critical('=================')

    # load dataset
    x_tr, y_tr, x_val, y_val, x_test, y_test = load_data(parameters['data_path'])

    print(x_tr,y_tr,x_val,y_val)
    print(len(x_tr),len(y_tr))

    import sys

    sys.path.append('..')
    import sshap
    from sklearn.svm import SVC

    # sshap.reproduce(10)
    model = SVC(decision_function_shape='ovo')
    idxes = list(np.arange(len(y_tr)))

    perf_runs = []
    sshap.reproduce(seed=42)
    np.random.shuffle(idxes)
    num_shards = 5
    num_instance_in_shard = len(y_tr) // num_shards
    ls = sshap.ShardedStruct(depth=1, nl=[idxes[:num_instance_in_shard], idxes[num_instance_in_shard:num_instance_in_shard*2],
                              idxes[num_instance_in_shard*2:num_instance_in_shard*3], idxes[num_instance_in_shard*3:num_instance_in_shard*4], 
                              idxes[num_instance_in_shard*4:num_instance_in_shard*5]])
    for c_run in range(runs):
        sshap.reproduce(seed=42 + c_run)       
        # ls = sshap.ShardedStruct(depth=1, nl=[idxes[:len(y_tr) // 3], idxes[len(y_tr) // 3:len(y_tr) // 3 * 2],
        #                                     idxes[len(y_tr) // 3 * 2:]])

        # ls = sshap.ShardedStruct(depth=1, nl=[idxes[:len(y_tr) // 2], idxes[len(y_tr) // 2:]])
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

        np.savez('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_car_evaluation_val.npz', loo=loo, rand_val=rand_val, sv=sv,
                 ssv=ssv, beta161=beta161, beta41=beta41,
                 beta14=beta14, beta116=beta116)
        np.savez('./result/' + t_str + f'_{str(c_run)}_{str(runs)}_' + 'car_evaluation_val.npz', loo=loo, rand_val=rand_val,
                 sv=sv, ssv=ssv, beta161=beta161, beta41=beta41,
                 beta14=beta14, beta116=beta116)

        # check the perf on test dataset
        perf_lists = []
        vales = np.load('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_car_evaluation_val.npz')
        for j, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
            perf_lists.append([])
            l2h = np.argsort(vales[alg])
            for i in range(len(y_tr), 0, -1):
                tmp_ls = copy.deepcopy(ls)
                tmp_ls.idxes_available = l2h[:i]
                acc = sshap.eval_utility(x_tr, y_tr, x_test, y_test, model, tmp_ls)
                perf_lists[j].append(acc)
        np.savetxt('./result/' + f'{str(c_run)}_{str(runs)}_' + 'latest_perf_list.txt', np.asarray(perf_lists))
        np.savetxt('./result/' + t_str + f'_{str(c_run)}_{str(runs)}_' + 'perf_list.txt', np.asarray(perf_lists))
        perf_runs.append(perf_lists)
    np.save('./result/' + 'latest_perf_runs.npy', np.asarray(perf_runs))
    np.save('./result/' + t_str + 'perf_runs.npy', np.asarray(perf_runs))

