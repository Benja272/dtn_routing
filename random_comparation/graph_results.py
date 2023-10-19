import os
import json
from typing import List
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt

DPI = 300
NETWORKS_PATH = './results/networks/'
COPIES_INDEX = 0
METRICS_INDEX = 1

def get_metrics_info():
    metrics = {}
    for net_name in os.listdir(NETWORKS_PATH):
        for copies_folder in os.listdir(NETWORKS_PATH + net_name):
            if copies_folder.startswith('copies='):
                copies = int(copies_folder[7:])
                for algorithm in os.listdir(NETWORKS_PATH + net_name + '/' + copies_folder):
                    path_to_algorithm = NETWORKS_PATH + net_name + '/' + copies_folder + '/' + algorithm
                    if(os.path.isdir(path_to_algorithm)):
                        print('reading ' + path_to_algorithm)
                        for folder in os.listdir(path_to_algorithm):
                            if 'metrics' == folder:
                                path_to_metrics = path_to_algorithm + '/metrics/'
                                if net_name not in metrics.keys():
                                    metrics[net_name] = {}
                                metrics[net_name][algorithm] = [copies, path_to_metrics]
    return metrics

def get_graph_data(nets_info):
    networks_info = {}
    print(nets_info)
    for net_name in nets_info.keys():
        for algotithm, m_info in nets_info[net_name].items():
            if net_name not in networks_info.keys():
                networks_info[net_name] = {}
            if m_info[COPIES_INDEX] not in networks_info[net_name].keys():
                networks_info[net_name][m_info[COPIES_INDEX]] = {}
            if algotithm not in networks_info[net_name][m_info[COPIES_INDEX]].keys():
                networks_info[net_name][m_info[COPIES_INDEX]][algotithm] = {}
            for metric in os.listdir(m_info[METRICS_INDEX]):
                print("reading " + m_info[METRICS_INDEX] + metric)
                if metric.endswith('.txt'):
                    with open(m_info[METRICS_INDEX] + metric, 'r') as file:
                        metric = metric.replace('.txt', '')
                        metric = metric.replace('METRIC=', '')
                        networks_info[net_name][m_info[COPIES_INDEX]][algotithm][metric] = literal_eval(file.read())
    return networks_info

def plot_comparation_graphs(network: str, copies_number: int):
    metrics_info = get_metrics_info()
    graph_data = get_graph_data(metrics_info)
    if network not in graph_data.keys():
        print('No such network')
        return
    if copies_number not in graph_data[network].keys():
        print('No such copies')
        return

    algoritms = list(graph_data[network][copies_number].keys())
    for metric in graph_data[network][copies_number][algoritms[0]].keys():
        plt.clf()
        # plt.figure(figsize=(1100/DPI, 800/DPI), dpi=DPI)
        plt.xticks(np.linspace(0, 1, 11))
        plt.title(metric)
        plt.xlabel('Failure probability')
        max_y = 0
        for algorithm in algoritms:
            data = list(zip(*graph_data[network][copies_number][algorithm][metric]))
            plt.plot(data[0], data[1], label=algorithm)
            max_y = max(max_y, max(data[1]))
        graphs_folder = './results/graphs/' + network + '/'
        plt.yticks(np.linspace(0, max_y, 10))
        plt.legend()
        create_folder(graphs_folder)
        plt.savefig(graphs_folder + metric + '.png', format='png', dpi=DPI)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

plot_comparation_graphs("rrn_start_t:7200,end_t:10800", 2)
plot_comparation_graphs("rrn_start_t:7200,end_t:14400", 2)
plot_comparation_graphs("rrn_start_t:10800,end_t:14400", 2)
plot_comparation_graphs("rrn_start_t:21600,end_t:28800", 2)

