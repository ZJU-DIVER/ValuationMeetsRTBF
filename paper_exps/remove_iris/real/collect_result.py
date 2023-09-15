import numpy as np
import configparser

# read parameters from config file
config = configparser.ConfigParser()
config.read('../config.ini')

p_default = config['DEFAULT']
runs = p_default.getint('RunningTimes')

participants = 100  # actually is 100, for draw use 100
choices = np.arange(50) * (participants // 100)  # chose 50%

data = np.load('../result/' + 'latest_perf_runs.npy')

for j, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
    # res = []
    # for i in range(runs):
    #     res.append(data[i][j][choices])
    # res = np.asarray(res)
    # np.savetxt(f'{alg}_acc.txt', res)

    wad = 0
    coef = 1
    for i in range(90):
        if i == 90 - 1:
            drop = np.average(data[0][j][i])
        else:
            drop = np.average(data[0][j][i]) - np.average(data[0][j][i+1])
        wad += coef * drop
        coef *= 0.95
    print(alg, wad)