import random
DTNSIM_PATH = "/home/benja/Documents/facu/tesis/project/dtnsim/dtnsim/src"
SEED = 10
PATH_TO_RESULT = 'results'

#########
RRN_TS_DURATION = 60
DATA_RATE = 12500
RRN_TS_DURATION_IN_SECONDS = 60

GROUND_TARGETS_EIDS = [x for x in range(7, 32) if x not in [11, 17, 28]]
SATELLITES = list(range(32, 48))
GS_EID = 1
F_RENAME_DTNSIM = dict([(GROUND_TARGETS_EIDS[i], i + 1) for i in range(len(GROUND_TARGETS_EIDS))]
                       + [(SATELLITES[i - len(GROUND_TARGETS_EIDS)], i + 1) for i in
                          range(len(GROUND_TARGETS_EIDS), len(GROUND_TARGETS_EIDS) + len(SATELLITES))]
                       + [(GS_EID, len(GROUND_TARGETS_EIDS) + len(SATELLITES) + 1)]
                       )
RRN_CP_PATH = "RingRoad_16sats_Walker_6hotspots_simtime24hs_comrange1000km.txt"
RRN_NUM_OF_REPS_OMNET = 100
###############
SOURCES = range(8) # Nodes that will send packages to every target (except themselves)
TARGETS = range(8) # Nodes that will recieve packages from every sources
NET_RNG = [0,1] #range(10)
RANDOM_TS_DURATION_IN_SECONDS = 10 # Duration of a single time stamp
RANDOM_NUM_OF_REPS = 1000 # Number of simulations that will be runned in OMNET++ for BRUF/IBRUF simulation
RANDOM_TRAFIC = dict((s+1, [t+1 for t in TARGETS if t != s]) for s in SOURCES)
###############

UTILS_PATH = '/home/benja/Documents/facu/tesis/project/comparation/utils'
SEED = 10
PATH_TO_RESULT = 'results'
random.seed(SEED)
COPIES_RNG = [1, 2, 3]
PF_RNG = [x / 100. for x in range(0, 110, 10)]

