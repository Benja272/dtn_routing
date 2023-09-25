import sys
import time
import resource
import ruting_morucop

priorities = [0,1,2]
file = 'simple_case.txt'
bundle_size = 1
max_copies = 2
time_start = time.perf_counter()
args = sys.argv
if (len(args) > 1):
    for i in range(1, len(args)):
        s = args[i].split("=")
        option = s[0]
        value = "".join(s[1:])
        if "--priorities" == option:
            priorities = [int(x) for x in value]
        elif "--contact_plan" == option:
            file = value
        elif "--bundle_size" == option:
            bundle_size = int(value)
        elif "--max_copies" == option:
            max_copies = int(value)
        else:
            print("Bad Option ", option)
            print("The options are: --priorities=[0,1,2] --contact_plan=filename.txt --bundle_size=1 --max_copies=2")
            exit()

net = ruting_morucop.Network.from_contact_plan("./use_cases/" + file, priorities=priorities, ts_duration=10)

print("Time on creating network: ", time.perf_counter() - time_start)

time_start = time.perf_counter()
#run your code
net.run_multiobjective_derivation(bundle_size, max_copies)
net.print_table()
time_elapsed = (time.perf_counter()- time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("%5.1f n_secs %5.1f MByte" % (time_elapsed,memMb))

net.export_rute_table([5,3])
