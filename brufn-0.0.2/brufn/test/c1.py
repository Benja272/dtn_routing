# Simple network with 3 contacts

NUM_OF_NODES = 3
CONTACTS = [{'from': 0 ,'to': 1, 'ts': 0, 'pf':0.1},
            {'from': 1 ,'to': 2, 'ts': 0, 'pf':0.1},
            {'from': 1,'to': 2, 'ts': 1, 'pf':0.5},
            {'from': 0,'to': 2, 'ts': 2, 'pf':0.5},
            {'from': 1,'to': 2, 'ts': 2, 'pf':0.5}
            ]
TRAFFIC = {'from': 0, 'to': 2, 'ts': 0, 'copies': 2}

