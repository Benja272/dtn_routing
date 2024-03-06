import os
from utils.parametric_compute_metrics import main

nets = ["net0", "net1", "rrn_start_t:0,end_t:10800", "rrn_start_t:7200,end_t:10800", "rrn_start_t:7200,end_t:14400", "rrn_start_t:10800,end_t:14400", "rrn_start_t:21600,end_t:28800", "rrn_start_t:21600,end_t:32400"]
algorithms = ["MORUCOP", "IRUCoPn"]

for net in nets:
  for c in [1,2,3]:
    for algorithm in algorithms:
      net_path = os.path.join("results/networks/", net, f"copies={c}", algorithm)
      reps = 10
      if net.startswith("rrn"):
        reps = 100
      main("/home/benja/Documents/facu/tesis/project/comparation", net_path, "", reps, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])