import network
import numpy as np

net = network.Network.from_contact_plan('simple_case.txt')
net.rucop(max_copies=2)
net.print_table()
