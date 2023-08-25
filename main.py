import network
import sys

priorities = None
if (len(sys.argv) > 1):
    arg = list(sys.argv[1])
    priorities = [int(x) for x in arg]
    net = network.Network.from_contact_plan('simple_case.txt', priorities=priorities)
else:
    net = network.Network.from_contact_plan('simple_case.txt')

net.rucop(max_copies=2)
net.print_table()
