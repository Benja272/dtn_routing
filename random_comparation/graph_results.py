import os
import json
from ast import literal_eval
from matplotlib import pyplot as plt

NET_INDEX = 0
COPIES_INDEX = 1
ALGORITHM_INDEX = 2
METRICS_INDEX = 3

def get_metrics_info():
    metrics = []
    for file_path in os.listdir('./results'):
        if file_path.startswith('net'):
            net_number: int = int(file_path[3:])
            for file in os.listdir('./results/' + file_path):
                if file.startswith('copies='):
                    copies = int(file[7:])
                    for algorithm in os.listdir('./results/' + file_path + '/' + file):
                        path_to_algorithm = './results/' + file_path + '/' + file + '/' + algorithm
                        for folder in os.listdir(path_to_algorithm):
                            if 'metrics' == folder:
                                path_to_metrics = path_to_algorithm + '/metrics'
                                metrics.append([net_number, copies, algorithm, path_to_metrics])
    return metrics

def get_graph_data():
    networks_info = {}
    metrics_info = get_metrics_info()
    for m_info in metrics_info:
        if m_info[NET_INDEX] not in networks_info.keys():
            networks_info[m_info[NET_INDEX]] = {}
        if m_info[COPIES_INDEX] not in networks_info[m_info[NET_INDEX]].keys():
            networks_info[m_info[NET_INDEX]][m_info[COPIES_INDEX]] = {}
        if m_info[ALGORITHM_INDEX] not in networks_info[m_info[NET_INDEX]][m_info[COPIES_INDEX]].keys():
            networks_info[m_info[NET_INDEX]][m_info[COPIES_INDEX]][m_info[ALGORITHM_INDEX]] = {}
        for metric in os.listdir(m_info[METRICS_INDEX]):
            with open(m_info[METRICS_INDEX] + '/' + metric, 'r') as file:
                metric = metric.replace('.txt', '')
                metric = metric.replace('METRIC=', '')
                networks_info[m_info[NET_INDEX]][m_info[COPIES_INDEX]][m_info[ALGORITHM_INDEX]][metric] = literal_eval(file.read())
    return networks_info

def plot_graph(metric_amount, network_number: int, copies_number: int, algorithm: str):
    graph_data = get_graph_data()
    print(graph_data)
    if network_number not in graph_data.keys():
        print('No such network')
        return
    if copies_number not in graph_data[network_number].keys():
        print('No such copies')
        return
    if algorithm not in graph_data[network_number][copies_number].keys():
        print('No such algorithm')
        return
    if metric_amount > len(graph_data[network_number][copies_number][algorithm].keys()):
        print('No such amount of metrics')
        return

    figure, axis = plt.subplots(1, metric_amount)
    for metric_index, metric  in enumerate(graph_data[network_number][copies_number][algorithm].keys()):
        if metric_index >= metric_amount:
            break
        data = list(zip(*graph_data[network_number][copies_number][algorithm][metric]))
        axis[metric_index].plot(data[0], data[1])
        axis[metric_index].set_title(metric)
        # plot graph

    plt.show()


plot_graph(6, 0, 2, 'IRUCoPn')
plot_graph(6, 0, 2, 'MORUCOP')

