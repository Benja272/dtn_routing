import sys
import time
import resource
import network

priorities = [1,2,3]
file = 'simple_case.txt'
bundle_size = 1
max_copies = 2
time_start = time.perf_counter()
args = sys.argv
if (len(args) > 1):
    for i in range(1, len(args)):
        option, value = args[i].split("=")
        match option:
            case "--priorities":
                priorities = [int(x) for x in value]
            case "--contact_plan":
                file = value
            case "--bundle_size":
                bundle_size = int(value)
            case "--max_copies":
                max_copies = int(value)
            case default:
                print("Bad Option ", option)
                exit()

net = network.Network.from_contact_plan("./use_cases/" + file, priorities=priorities)

print("Time on creating network: ", time.perf_counter() - time_start)

time_start = time.perf_counter()
#run your code
net.run_multiobjective_derivation(bundle_size, max_copies)
net.print_table()
time_elapsed = (time.perf_counter()- time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("%5.1f n_secs %5.1f MByte" % (time_elapsed,memMb))

net.export_rute_table([5])
