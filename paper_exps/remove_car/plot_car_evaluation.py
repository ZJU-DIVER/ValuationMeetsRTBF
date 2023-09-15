import matplotlib.pyplot as plt
import numpy as np
import configparser
import os
import time

def current_dir():
    return os.path.dirname(os.path.realpath(__file__))


def read_config_file(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_plot = config['PLOT']
    parameters['data'] = current_dir() + os.sep + p_plot['Data']
    parameters['Y'] = p_plot['Y']
    parameters['Percent'] = int(p_plot['Percent'])
    return parameters


p = read_config_file('config.ini')

# set plot fig
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
Y = p["Y"]

# draw 50 point, remove
x = np.arange(0, p['Percent'])

# set labels
axes.set_xlabel("Fraction of train data removed (%)", fontsize='large')
axes.set_ylabel("Prediction accuracy", fontsize='large')
axes.set_title(f"Dataset: Car Evaluation", fontsize='x-large')
# axes.xaxis.labelpad = 10

labels = {
    'rand_val': 'Random',
    'loo': 'LOO',
    'sv': 'SV',
    'ssv': 'ssv',
    'beta161': 'Beta(16, 1)',
    'beta41': 'Beta(4, 1)',
    'beta14': 'Beta(1, 4)',
    'beta116': 'Beta(1, 16)'
}

# beta use the sample
linestyle_tuple = [
    ('solid', (0, ())),
    ('densely dotted', (0, (1, 1))),

    ('densely dashed', (0, (5, 1))),
    ('densely dashed', (0, (5, 1))),
    ('densely dashed', (0, (5, 1))),
    ('densely dashed', (0, (5, 1))),

    ('densely dashdotted', (0, (3, 1, 1, 1))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

    ('dotted', (0, (1, 1))),
    ('solid', (0, ())),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('loosely dotted', (0, (1, 10))),
    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

# Real plot here
for i, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
    data = np.loadtxt(p['data'] + os.sep + f'{alg}_{Y}.txt')
    data = data[:, :p['Percent']]
    # Add lines in the fig
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if alg == 'beta116' or alg == 'beta14':
        continue
    else:
        axes.plot(x, mean, linestyle=linestyle_tuple[i][-1], label=labels[alg])
        # Add the shadow
        runs = len(data)
        axes.fill_between(x, mean - std / np.sqrt(runs - 1), mean + std / np.sqrt(runs - 1), alpha=0.2)

plt.legend()
plt.savefig(f'latest_car_evaluation.pdf', bbox_inches='tight')


t_str = time.strftime("%Y%m%d%H%M%S")
plt.savefig(f'{t_str}_car_evaluation.pdf', bbox_inches='tight')

