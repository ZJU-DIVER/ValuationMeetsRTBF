import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import configparser
import os
import time

matplotlib.rcParams['mathtext.fontset'] = 'cm'  # 设置数学模式字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置画图字体
custom_size = 16

o_c = [(219, 83, 117), (183, 153, 156), (170, 170, 170), (114, 147, 160), (223, 190, 153), (85, 98, 112)] # , (110, 201, 195), (187, 197, 211)

colors = []
for o in o_c:
    t = '#'
    for v in o:
        t += str(hex(v))[-2:]
    colors.append(t)

print(colors)

font1 = {'family': 'Arial',
         'weight': 'bold',
         'size': custom_size,
         }

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
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
Y = p["Y"]

# draw 50 point, remove
x = np.arange(0, p['Percent'])

# set labels
axes.set_xlabel("Fraction of train data removed (%)", font1)
axes.set_ylabel("Prediction accuracy", font1)
axes.tick_params(axis='y', which='major', labelsize=custom_size)
axes.tick_params(axis='x', which='major', labelsize=custom_size, rotation=45)
# axes.tick_params(axis='both', which='minor', labelsize=8)
# axes.set_title(f"Dataset: Iris", fontsize='x-large')
# axes.xaxis.labelpad = 10

labels = {
    'rand_val': 'Random',
    'loo': 'LOO',
    'sv': 'SV',
    'ssv': 'SSV',
    'beta161': 'Beta(16,1)',
    'beta41': 'Beta(4, 1)',
    'beta14': 'Beta(1, 4)',
    'beta116': 'Beta(1,16)'
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


markers = [
    'o', 's', 'v', '^', 'D', 'x'
]

# Real plot here
cnt = 0
for i, alg in enumerate(['ssv', 'sv', 'beta161', 'beta41', 'beta116', 'beta14', 'loo', 'rand_val']):
    data = np.loadtxt(p['data'] + os.sep + f'{alg}_{Y}.txt')
    data = data[:, :p['Percent']]
    # Add lines in the fig
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if alg == 'beta14' or alg == 'beta41':
        continue
    else:
        axes.plot(x, mean, linestyle=linestyle_tuple[i][-1], label=labels[alg], linewidth=4, color=colors[cnt],  zorder=10-i) # marker=markers[cnt], markevery=10, markeredgewidth=1.5, markerfacecolor='none',
        n_drop = len(mean)
        wad = 0
        coef = 1
        for drop_idx in range(n_drop):
            if drop_idx == n_drop - 1:
                drop = mean[drop_idx]
            else:
                drop = mean[drop_idx] - mean[drop_idx+1]
            # wad += (n_drop-1-drop_idx) / (n_drop) * drop
            wad += coef * drop
            coef *= 0.95
        print(f'alg: {alg}\t wad: {wad}')
        # Add the shadow
        runs = len(data)
        axes.fill_between(x, mean - std / np.sqrt(runs - 1), mean + std / np.sqrt(runs - 1), alpha=0.2, color=colors[cnt])
        cnt += 1

for k, spine in axes.spines.items():  # ax.spines is a dictionary
    spine.set_zorder(10)
    spine.set_linewidth(2)

axes.margins(x=0.03)

from matplotlib.image import imread
from tempfile import NamedTemporaryFile

# def get_size(fig, dpi=100):
#     with NamedTemporaryFile(suffix='.png') as f:
#         fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
#         height, width, _channels = imread(f.name).shape
#         return width / dpi, height / dpi
#
# def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
#     target_width, target_height = size
#     set_width, set_height = target_width, target_height # reasonable starting point
#     deltas = [] # how far we have
#     while True:
#         fig.set_size_inches([set_width, set_height])
#         actual_width, actual_height = get_size(fig, dpi=dpi)
#         set_width *= target_width / actual_width
#         set_height *= target_height / actual_height
#         deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
#         if deltas[-1] < eps:
#             return True
#         if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
#             return False
#         if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
#             return False
#
# set_size(fig, (5, 4))
plt.legend(prop={'size': 16}).set_zorder(12)
# plt.legend(ncol = 6,bbox_to_anchor=(0.18, 1.02, 0.82, 0.2), loc="lower left", frameon=False, prop={'size': 14})  # ncol = 2 fontsize='x-small'
plt.savefig(f'latest_iris.pdf', bbox_inches='tight')


t_str = time.strftime("%Y%m%d%H%M%S")
plt.savefig(f'{t_str}_iris.pdf', bbox_inches='tight')

