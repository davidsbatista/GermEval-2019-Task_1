# coding: utf-8
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

losses = []
with open('./loss.txt', 'r') as fh:
    epoch_end = False
    for i_line, line in enumerate(fh):
        t, i_step, l, *tail = line.strip().split('\t')
        if t == 'epoch' and tail[-1] == 'train':
            epoch_end = True
            continue
        loss = {'i_step': i_line, 'loss': float(l), 'type': tail[-1], 'epoch_end': epoch_end}
        epoch_end = False
        losses.append(loss)
    df = pd.DataFrame.from_records(losses).set_index('i_step')
    print(df.loc[df.type == 'test'].head())
    fig, ax = plt.subplots(1, 1)
    ax = df.loc[df['type'] == 'train'].loss.plot(ax=ax, alpha=0.15)
    ax = df.loc[df['type'] == 'train'].loss.rolling(2).mean().plot(ax=ax, color='b', alpha=0.75)
    ax.vlines(np.where(df['type'] == 'test')[0], ymin=0, ymax=df.loss.loc[df.type == 'test'], color='g', alpha=0.5)
    epochs = np.where(df.epoch_end)[0]
    ax.vlines(epochs, ymin=0, ymax=df.loss.max(), colors='r', alpha=0.25, linestyles='--')
    plt.savefig('./loss.png')
    plt.close()
    
