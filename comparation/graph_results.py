import os
import json
from typing import List
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt
import ipdb

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
                        for folder in os.listdir(path_to_algorithm):
                            if 'metrics' == folder:
                                path_to_metrics = path_to_algorithm + '/metrics/'
                                if net_name not in metrics.keys():
                                    metrics[net_name] = {}
                                if algorithm not in metrics[net_name].keys():
                                    metrics[net_name][algorithm] = {}
                                metrics[net_name][algorithm][copies] = path_to_metrics
    return metrics

def get_graph_data(nets_info):
    networks_info = {}
    for net_name in nets_info.keys():
        for algorithm, m_info in nets_info[net_name].items():
            if net_name not in networks_info.keys():
                networks_info[net_name] = {}
            for copies, path_to_metrics in m_info.items():
                if copies not in networks_info[net_name].keys():
                    networks_info[net_name][copies] = {}
                if algorithm not in networks_info[net_name][copies].keys():
                    networks_info[net_name][copies][algorithm] = {}
                for metric in os.listdir(path_to_metrics):
                    if metric.endswith('.txt'):
                        with open(path_to_metrics + metric, 'r') as file:
                            metric = metric.replace('.txt', '')
                            metric = metric.replace('METRIC=', '')
                            networks_info[net_name][copies][algorithm][metric] = literal_eval(file.read())
    return networks_info

def plot_comparation_graphs(graph_data, networks):
    for network in networks:
        if network not in graph_data.keys():
            print('No such network')
            continue

        for copies_number in graph_data[network].keys():
            algoritms = list(graph_data[network][copies_number].keys())
            print("Comparation of algoritms: ", algoritms, "for network ", network, "with ", copies_number, "copies")
            for metric in graph_data[network][copies_number][algoritms[0]].keys():
                plt.clf()
                title = metric
                if metric == 'deliveryRatio': title = 'Delivery Ratio'
                if metric == 'appBundleReceivedDelay:mean': title = 'Delay Promedio'
                if metric == 'EnergyEfficiency': title = 'Eficiencia Energetica'

                plt.title(title)
                plt.xlabel('Probabilidad de Fallo')
                max_y = 0
                for algorithm in algoritms:
                    data = graph_data[network][copies_number][algorithm][metric]
                    # positions = list(zip(*data))[0]
                    # positions = tuple(float(i) for i in positions)
                    positions1 = [i for i in range(1, 23,2)]
                    positions2 = [i for i in range(2, 23,2)]
                    boxes = list(map(lambda x: {
                        'med': x[1],
                        'q1': x[1] - x[2],
                        'q3': x[1] + x[2],
                        'whislo': x[4],
                        'whishi': x[3]
                    }, data))
                    axs = plt.axes()
                    pf_rng = list(zip(*data))[0]
                    if algorithm == 'IRUCoPn':
                        algorithm = 'RUCoP'
                        axs.bxp(boxes, positions1, showfliers=False,
                            medianprops={'color':'orange', 'linewidth':1.5}, boxprops={'color':'orange'}, widths=0.3)
                        plt.plot(np.NaN, np.NaN, color='orange', label='RUCoP')
                        # plt.plot(data[0], data[1], label=algorithm, color='orange')
                    else:
                        axs.bxp(boxes, positions2, showfliers=False,
                            medianprops={'color':'tab:blue', 'linewidth':1.5}, boxprops={'color':'tab:blue'}, widths=0.3)
                        plt.plot(np.NaN, np.NaN, color='tab:blue', label=algorithm)
                        # plt.plot(data[0], data[1], label=algorithm, color='tab:blue')
                    # max_y = max(max_y, max(data[1]))
                axs.set_xticklabels(pf_rng)
                axs.set_xticks(tuple(1.5 + 2*i for i in range(len(pf_rng))))
                axs.set_xlim(0, 23)
                # draw temporary red and blue lines and use them to create a legend
                graphs_folder = './results/graphs/' + network + '/copies=' + str(copies_number) + '/'
                # plt.yticks(np.linspace(0, max_y, 10))
                plt.legend()
                create_folder(graphs_folder)
                plt.savefig(graphs_folder + metric + '.png', format='png', dpi=DPI)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


metrics_info = get_metrics_info()
graph_data = get_graph_data(metrics_info)
# plot_comparation_graphs(graph_data, graph_data.keys())
plot_comparation_graphs(graph_data, ['net1'])

