NUM_OF_NODES = 8
CONTACTS = [{'from': 0, 'to': 1, 'ts': 0}, {'from': 0, 'to': 4, 'ts': 0}, {'from': 2, 'to': 3, 'ts': 0}, {'from': 2, 'to': 4, 'ts': 0}, {'from': 3, 'to': 0, 'ts': 0}, {'from': 3, 'to': 7, 'ts': 0}, {'from': 4, 'to': 0, 'ts': 0}, {'from': 4, 'to': 7, 'ts': 0}, {'from': 6, 'to': 1, 'ts': 0}, {'from': 7, 'to': 2, 'ts': 0}, {'from': 7, 'to': 4, 'ts': 0}, {'from': 0, 'to': 5, 'ts': 1}, {'from': 0, 'to': 6, 'ts': 1}, {'from': 2, 'to': 1, 'ts': 1}, {'from': 2, 'to': 5, 'ts': 1}, {'from': 3, 'to': 1, 'ts': 1}, {'from': 4, 'to': 0, 'ts': 1}, {'from': 4, 'to': 1, 'ts': 1}, {'from': 4, 'to': 5, 'ts': 1}, {'from': 5, 'to': 7, 'ts': 1}, {'from': 7, 'to': 0, 'ts': 1}, {'from': 7, 'to': 5, 'ts': 1}, {'from': 0, 'to': 1, 'ts': 2}, {'from': 0, 'to': 3, 'ts': 2}, {'from': 0, 'to': 7, 'ts': 2}, {'from': 2, 'to': 6, 'ts': 2}, {'from': 3, 'to': 7, 'ts': 2}, {'from': 6, 'to': 1, 'ts': 2}, {'from': 6, 'to': 3, 'ts': 2}, {'from': 6, 'to': 4, 'ts': 2}, {'from': 7, 'to': 2, 'ts': 2}, {'from': 7, 'to': 3, 'ts': 2}, {'from': 7, 'to': 4, 'ts': 2}, {'from': 0, 'to': 4, 'ts': 3}, {'from': 0, 'to': 7, 'ts': 3}, {'from': 1, 'to': 4, 'ts': 3}, {'from': 1, 'to': 5, 'ts': 3}, {'from': 2, 'to': 0, 'ts': 3}, {'from': 2, 'to': 6, 'ts': 3}, {'from': 4, 'to': 2, 'ts': 3}, {'from': 4, 'to': 7, 'ts': 3}, {'from': 6, 'to': 2, 'ts': 3}, {'from': 6, 'to': 5, 'ts': 3}, {'from': 7, 'to': 0, 'ts': 3}, {'from': 1, 'to': 0, 'ts': 4}, {'from': 1, 'to': 2, 'ts': 4}, {'from': 1, 'to': 6, 'ts': 4}, {'from': 3, 'to': 7, 'ts': 4}, {'from': 4, 'to': 2, 'ts': 4}, {'from': 4, 'to': 3, 'ts': 4}, {'from': 4, 'to': 5, 'ts': 4}, {'from': 4, 'to': 7, 'ts': 4}, {'from': 5, 'to': 0, 'ts': 4}, {'from': 6, 'to': 2, 'ts': 4}, {'from': 7, 'to': 4, 'ts': 4}, {'from': 0, 'to': 1, 'ts': 5}, {'from': 0, 'to': 3, 'ts': 5}, {'from': 0, 'to': 6, 'ts': 5}, {'from': 2, 'to': 5, 'ts': 5}, {'from': 2, 'to': 7, 'ts': 5}, {'from': 3, 'to': 0, 'ts': 5}, {'from': 5, 'to': 4, 'ts': 5}, {'from': 5, 'to': 6, 'ts': 5}, {'from': 6, 'to': 5, 'ts': 5}, {'from': 6, 'to': 7, 'ts': 5}, {'from': 7, 'to': 1, 'ts': 5}, {'from': 0, 'to': 2, 'ts': 6}, {'from': 0, 'to': 4, 'ts': 6}, {'from': 1, 'to': 3, 'ts': 6}, {'from': 2, 'to': 6, 'ts': 6}, {'from': 3, 'to': 7, 'ts': 6}, {'from': 4, 'to': 6, 'ts': 6}, {'from': 5, 'to': 1, 'ts': 6}, {'from': 5, 'to': 4, 'ts': 6}, {'from': 6, 'to': 0, 'ts': 6}, {'from': 6, 'to': 7, 'ts': 6}, {'from': 7, 'to': 1, 'ts': 6}, {'from': 0, 'to': 1, 'ts': 7}, {'from': 0, 'to': 3, 'ts': 7}, {'from': 1, 'to': 0, 'ts': 7}, {'from': 2, 'to': 7, 'ts': 7}, {'from': 3, 'to': 6, 'ts': 7}, {'from': 4, 'to': 1, 'ts': 7}, {'from': 4, 'to': 6, 'ts': 7}, {'from': 5, 'to': 3, 'ts': 7}, {'from': 6, 'to': 0, 'ts': 7}, {'from': 6, 'to': 5, 'ts': 7}, {'from': 7, 'to': 6, 'ts': 7}, {'from': 0, 'to': 2, 'ts': 8}, {'from': 0, 'to': 6, 'ts': 8}, {'from': 2, 'to': 3, 'ts': 8}, {'from': 2, 'to': 6, 'ts': 8}, {'from': 3, 'to': 2, 'ts': 8}, {'from': 3, 'to': 4, 'ts': 8}, {'from': 4, 'to': 3, 'ts': 8}, {'from': 4, 'to': 5, 'ts': 8}, {'from': 6, 'to': 3, 'ts': 8}, {'from': 6, 'to': 4, 'ts': 8}, {'from': 6, 'to': 7, 'ts': 8}, {'from': 0, 'to': 5, 'ts': 9}, {'from': 0, 'to': 6, 'ts': 9}, {'from': 1, 'to': 2, 'ts': 9}, {'from': 1, 'to': 7, 'ts': 9}, {'from': 4, 'to': 0, 'ts': 9}, {'from': 5, 'to': 0, 'ts': 9}, {'from': 5, 'to': 4, 'ts': 9}, {'from': 6, 'to': 1, 'ts': 9}, {'from': 6, 'to': 3, 'ts': 9}, {'from': 7, 'to': 0, 'ts': 9}, {'from': 7, 'to': 3, 'ts': 9}] 
