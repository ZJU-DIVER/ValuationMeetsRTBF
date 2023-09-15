import numpy as np

runs = 3
participants = 1000  # actually is 100, for draw use 100 #100
choices = np.arange(50) * (participants // 100)  # chose 50%

data = np.load('../result/' + 'latest_perf_runs.npy')
print(data.shape)

for j, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
    res = []
    for i in range(runs):
        res.append(data[i][j][choices])
    res = np.asarray(res)
    np.savetxt(f'{alg}_acc.txt', res)

    wad = 0
    coef = 1
    for i in range(participants):
        if i == participants - 1:
            drop = np.average(data[2][j][i])
        else:
            drop = np.average(data[2][j][i]) - np.average(data[2][j][i+1])
        wad += coef * drop
        coef *= 0.95
    print(alg, wad)