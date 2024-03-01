'''
This script process a sample generated by DTNSIM and and computes plots for the followings metrics:
    -> Received Bundles in total
    -> Bundles Rerouted in total (amount of times that nodes re-route bundles)
    -> Hop counts in total (amount of hops that deliver bundles do in their path to destination)

All metrics will be ploted vs proportion of deleted contacts.

Arguments
    -> repetitions: Amount of repetitions for each contact plan. If for a contact plan there aren't more than one runs
                    it should be setted to 1.
    -> deletedContact: Maximun number of deleted contact. Input data must be contain a file for each case in [0
                       to deletedContact].

    -> INPUT_PATH: Path to folder that contains input files.
    -> OUTPUT_PATH: Path to folder in which script will write results.

Also, there is the following convention for input files (stored in INPUT_PATH/):

    dtnsim-faultsAware=%IS FAULT AWARE%,deleteNContacts=%NUMBER OF DELETED CONTACTS%-#%RUN NUMBER%.sca

The varriable parts in string are marked with %%. They are:
    -> %IS FAULT AWARE%
    -> %NUMBER OF DELETED CONTACTS%
    -> %RUN NUMBER%

OUTPUT:

"%s/METRIC=%s-FAULTAWARE=%s-MAX_DELETED_CONTACTS=%d-.txt"%(OUTPUT_PATH,metric,aware,MAX_DELETED_CONTACTS)

'''


#Received packet after a random atack, deleted contact/ number of contact
import sqlite3
import matplotlib.pyplot as plt
from functools import reduce
import sys
sys.path.append('../')
import os
import json
from statistics import mean, stdev
# from brufn.utils import getListFromFile
from settings import *
metrics_name = list(map(lambda m: m[0], METRICS))

def main(exp_path, net, routing_algotithm, num_of_reps, pf_rng):
        ra_dir = os.path.join(exp_path, net, routing_algotithm)
        pf_rng_str = pf_rng_to_str(pf_rng)
        graph_output_dir = os.path.join(ra_dir, 'metrics')
        os.makedirs(graph_output_dir, exist_ok=True)
        path = os.path.join(ra_dir, 'results', "dtnsim-faultsAware=false")
        metric_graph_data = {metric: [] for metric in metrics_name}
        for pf_str in pf_rng_str:
            metrics_sum = {metric: [] for metric in metrics_name}
            path_with_fp = path + f",failureProbability={pf_str}"
            for i in range(num_of_reps):
                path_with_rep_sca = path_with_fp + f"-#{i}.sca"
                path_with_rep_vec = path_with_fp + f"-#{i}.vec"
                print(path_with_rep_sca)
                cursor_sca = fileCursor(path_with_rep_sca)
                cursor_vec = fileCursor(path_with_rep_vec)
                for metric in metrics_name:
                    if(metric == "deliveryRatio"):
                        metrics_sum[metric].append(deliveryRatio(cursor_sca))
                    elif( (metric == "appBundleReceivedDelay:mean") or (metric == "appBundleReceivedHops:mean") or (metric == "sdrBundleStored:timeavg")):
                        metrics_sum[metric].append(executeOperation(cursor_sca, "AVG", metric))
                    elif metric=='EnergyEfficiency':
                        delivered_bundles = executeOperation(cursor_sca, "SUM", "appBundleReceived:count")
                        number_of_transmisions = executeOperation(cursor_sca, "SUM", "dtnBundleSentToCom:count")
                        metrics_sum[metric].append(getEnergyEfficiency(cursor_vec))
                        metrics_sum["appBundleReceived:count"].append(delivered_bundles)
                        metrics_sum["dtnBundleSentToCom:count"].append(number_of_transmisions)
                        # f_delivered_bundles =  getListFromFile(f"{graph_output_dir}/METRIC=appBundleReceived:count.txt")
                        # f_number_of_transmisions = getListFromFile(f"{graph_output_dir}/METRIC=dtnBundleSentToCom:count.txt")
                        # f_avg_by_rep = [(f_delivered_bundles[i][0], f_delivered_bundles[i][1] / f_number_of_transmisions[i][1] if f_number_of_transmisions[i][1] != 0 else 0) for i in range(len(f_delivered_bundles))]

                    #compute average function for all contact plans (all contact plans average - DENSITY AVERAGE)
            for metric in metrics_name:
                graph_data = (pf_str, mean(metrics_sum[metric]), stdev(metrics_sum[metric]), max(metrics_sum[metric]), min(metrics_sum[metric]))
                metric_graph_data[metric].append(graph_data)

        for metric in metrics_name:
            file = f"{graph_output_dir}/METRIC={metric}.txt"
            print(file)
            text_file = open(file, "w")
            text_file.write(str(metric_graph_data[metric]))
            text_file.close()


def pf_rng_to_str(pf_rng):
    res = []
    for pf in pf_rng:
        if pf == 0:
            res.append('0')
        elif pf == 1:
            res.append('1')
        else:
            res.append(str(pf))
    return res

def fileCursor(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return cur

def getEnergyEfficiency(cursor_vec):
    sentToComPkgs = vectorCount(cursor_vec, "dtnBundleSentToComIds:vector")
    arrivedToAppPkgs = vectorCount(cursor_vec, "dtnBundleSentToAppIds:vector")
    arrivedIds = list(arrivedToAppPkgs.keys())
    transmisions = 0
    for id in arrivedIds:
        transmisions += sentToComPkgs[id]
    return len(arrivedIds)/transmisions if transmisions != 0 else 1

def vectorCount(cur, vectorName):
    cur.execute("SELECT vectordata.value, COUNT(*) FROM vectordata INNER JOIN " +
                "vector ON vectordata.vectorId = vector.vectorId "+
                "WHERE vector.vectorName='%s' GROUP BY vectordata.value"%(vectorName))
    rows = dict(cur.fetchall())
    return rows


def executeOperation(cur, operation, scalarName):
    cur.execute("SELECT %s(scalarValue) AS result FROM scalar WHERE scalarName='%s'"%(operation, scalarName))
    rows = cur.fetchall()
    return 0 if (rows[0]["result"] == None) else float(rows[0]["result"])

def deliveryRatio(cursor):
    delivered_bundles = executeOperation(cursor, "SUM", "appBundleReceived:count")
    sent_bundles = executeOperation(cursor, "SUM", "appBundleSent:count")
    return delivered_bundles / sent_bundles if sent_bundles != 0 else 0


'''
Given a list of list of pairs: [[(x0,y0),...] , [(xn,yn),....]]
returns a unique list compute as average of above functions
'''
def promList(llist):
  assert len(llist) == len(CP_RANGE), "error amount of contact plans"
  assert len(list(filter(lambda l: len(l) != len([x/10 for x in range(11)]), llist))) == 0, "error failure probability"

  llist = [list(map(lambda f: f[x], llist)) for x in range(len([x/10 for x in range(11)]))]
  llist = list(map(lambda l: (reduce(lambda x,y: (x[0] + y[0],x[1] + y[1]),l), statistics.stdev([t[1] for t in l])), llist))
  return [(x[0][0]/float(len(CP_RANGE)), x[0][1]/float(len(CP_RANGE)), x[1]) for x in llist]


if __name__ == "__main__":
    if len(sys.argv) == 6:
        '''
            sys.argv[1] -> Experiment path
            sys.argv[2] -> net
            sys.argv[3] -> routing_algotithm
            sys.argv[4] -> num_of_reps
            sys.argv[5] -> pf_rng

        '''

        main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), json.loads(sys.argv[5]))
    else:
        print("Error: Invalid number of arguments")
        exit(-1)

