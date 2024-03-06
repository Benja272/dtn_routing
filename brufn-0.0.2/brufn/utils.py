from functools import reduce
from typing import List, Tuple
import ast
import os
import matplotlib.pyplot as plt
import numpy as np

# Color of ploted functions
COLORS = ['gray', 'blue', 'green', 'orange', 'red', 'darkmagenta', 'black'] * 3

# Style of plotted function
STYLES = ['--^', '--o', '--s', '--v', '--p', '--x', '--+', '--<', '-->', '--H', '--h',  ] * 3


def average(lst: List[float]):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def getListFromFile(path):
    ''''
    Get a list from file reading first line only
    '''

    with open(path) as f:
        lines = f.readlines()

    return ast.literal_eval(lines[0])


def print_str_to_file(data, output_path: str):
    f = open(output_path, 'w')
    f.write(data)
    f.close()


def getListFromFileWhenDic(path):
    with open(path) as f:
        lines = f.readlines()

    d = ast.literal_eval(lines[0])
    return sorted([(float(x), d[x]) for x in d.keys()], key=lambda x: x[0])


def prom_llist(llist: List[List[Tuple[float, float]]]) -> List[float]:
    assert len(llist) > 0 and all(len(l) == len(llist[0]) for l in llist)
    res = []
    for i in range(len(llist[0])):
        res.append((llist[0][i][0], sum(map(lambda x: x[i][1], llist)) / len(llist)))
    return res


def plot_function(f: List[Tuple[float, float]], output_path, f_label=' ', xlabel='', ylabel=' ', ftype='png'):
    line_up, = plt.plot([x[0] for x in f], [y[1] for y in f], '--o', label=f_label)
    plt.legend(handles=[line_up])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color='gray', linestyle='dashed')
    plt.savefig(output_path + '.' + ftype)
    plt.clf()
    plt.cla()
    plt.close()


def plot_bar(functions: List[Tuple[float, float]], output_path: str, labels: List[str] = [], xlabel=' ', ylabel=' ',
             ftype='png', colors=COLORS, bar_width=0.3, separation_width=0.3):
    fig, ax = plt.subplots()

    index = np.arange(len(functions[0])) * [(len(functions) + 1) * bar_width for p in range(0, len(functions[0]))]

    for f in range(len(functions)):
        label = labels[f] if len(labels) > f else ' '
        ax.bar(index + f * bar_width, [y[1] for y in functions[f]], bar_width, color=colors[f], label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticks =  index + [ (len(functions) * bar_width) / 2 for x in range(len(functions[0]))]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x[0]) for x in functions[0]])
    ax.legend()
    plt.savefig(output_path + '.' + ftype)
    plt.clf()
    plt.cla()
    plt.close()

def plot_nfunctions(functions: List[Tuple[float, float]], output_path: str, labels: List[str] = [], xlabel=' ',
                    ylabel=' ',ftype='png', colors=COLORS, styles=STYLES, **kwargs):
    if labels == []:
        labels = [' '] * len(functions)

    lines = []
    for f, name, color, style in zip(functions, labels, colors, styles):
        line, = plt.plot([x[0] for x in f], [y[1] for y in f], style, label=name, color=color)
        lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if 'yticks' in kwargs:
        yticks = kwargs['yticks']
        if type(yticks) != tuple or len(yticks) < 2 or len(yticks) > 3:
            raise ValueError("yticks must be a tuple of integers (from, to, step) or (from,to) leaving step unsetted")
        elif len(yticks) == 3:
            plt.yticks(np.arange(kwargs['yticks'][0], kwargs['yticks'][1], step=kwargs['yticks'][2]))
        elif len(yticks) == 2:
            plt.yticks(np.arange(kwargs['yticks'][0], kwargs['yticks'][1]))

    plt.grid(color='gray', linestyle='dashed')
    plt.savefig(output_path + '.' + ftype)
    plt.clf()
    plt.cla()
    plt.close()
