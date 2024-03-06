
import os
from brufn.net_metrics_generator import NetMetricGenerator
from brufn.contact_plan import ContactPlan
from brufn.network import Net, Contact
from copy import copy
from settings import *

def generate_RNN_net_from_cp(startt, endt):
    os.makedirs(PATH_TO_RESULT, exist_ok=True)
    step = (endt - startt) // 3600
    cp = ContactPlan.parse_contact_plan(RRN_CP_PATH)
    cp = cp.filter_contact_by_endpoints(GROUND_TARGETS_EIDS + SATELLITES, SATELLITES + [GS_EID])
    cp.rename_eids(F_RENAME_DTNSIM)

    slice_dir = os.path.join(PATH_TO_RESULT,f'networks/rrn_start_t:{startt},end_t:{endt}')
    dtnsim_cp_path = os.path.join(slice_dir, f'RRN_A_with_ISL_seed={SEED}_reflexive.dtnsim')
    os.makedirs(slice_dir, exist_ok=True)

    sliced_cp_6h = cp.slice_cp(startt, endt)
    sliced_cp_6h = sliced_cp_6h.shift_cp_in_time(-startt)
    sliced_cp_path = os.path.join(slice_dir, f'cp_start_t:{startt},end_t:{endt}.txt')
    sliced_cp_6h.print_to_file(sliced_cp_path)
    net = sliced_cp_6h.generate_slicing_abstraction(RRN_TS_DURATION_IN_SECONDS, 60)
    net.print_to_file(slice_dir)

    contacts_by_ts = dict((ts, []) for ts in range(net.num_of_ts))
    print("Number of contacts: ", len(net.contacts))
    for c in net.contacts:
        contacts_by_ts[c.ts].append(copy(c))
        c_simetric = Contact(c.to, c.from_, c.ts, pf=c.pf, begin_time=c.begin_time, end_time=c.end_time)
        if c_simetric not in net.contacts:
            print(f"add {c_simetric}")
            contacts_by_ts[c.ts].append(c_simetric)

    contacts = []
    for ts in range(net.num_of_ts):
        nodes = list(range(net.num_of_nodes))
        while len(nodes) > 0:
            node = nodes.pop(0)
            snd_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.from_], key=lambda c: c.to)
            rcv_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.to], key=lambda c: c.from_)
            # According to cp2modest.py all contacts must be full-duplex (reflexive) and only one full-duplex communication is allowed
            if len(snd_contacts) == len(rcv_contacts):
                if len(snd_contacts) > 0:
                    for i in random.sample(range(len(snd_contacts)), len(snd_contacts)):
                        if snd_contacts[i].to in nodes:
                            contacts.append(snd_contacts[i])
                            contacts.append(rcv_contacts[i])
                            nodes.remove(snd_contacts[i].to)
                            assert snd_contacts[i].to == rcv_contacts[i].from_, "must add a symmetrical contact"
                            break
            else:
                assert False, f"node {node} has unidirectional contact in slot {ts}"

        sorted(contacts, key=lambda x: (x.ts, x.from_, x.to))
        net = Net(net.num_of_nodes, [Contact(c.from_, c.to, c.ts) for c in contacts])
        net.print_to_file(slice_dir, file_name=f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
        net.print_dtnsim_cp_to_file(60, 100, dtnsim_cp_path)
        net.to_dot(slice_dir, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.dot')
        NetMetricGenerator(net, range(net.num_of_nodes), [1,2,3], [1,7,15], slice_dir).compute_metrics()

def generate_random_networks(net_rng, ts_duration_in_seconds):
    '''
    Generate a network with full duplex connection and at most 1 bidireccional contact per node.
    The network is obtained from one of the 10 random networks used in ICC
    '''
    for net in net_rng:
        path = os.path.join(PATH_TO_RESULT, 'networks', f'net{net}'); os.makedirs(path, exist_ok=True)
        path_to_net = os.path.join(path, f'net-{net}-seed={SEED}.py')
        dtnsim_cp_path = os.path.join(path, f'0.2_{net}_seed={SEED}_reflexive.dtnsim')
        if not os.path.isfile(path_to_net):
            net_obj = Net.get_net_from_file('10NetICC/net0/net.py', contact_pf_required=False)
            contacts_by_ts = dict((ts, []) for ts in range(net_obj.num_of_ts))
            print("Number of contacts: ", len(net_obj.contacts))
            for c in net_obj.contacts:
                contacts_by_ts[c.ts].append(copy(c))
                c_simetric = Contact(c.to, c.from_, c.ts, pf=c.pf, begin_time=c.begin_time, end_time=c.end_time)
                if c_simetric not in net_obj.contacts:
                    print(f"add {c_simetric}")
                    contacts_by_ts[c.ts].append(c_simetric)

            contacts = []
            for ts in range(net_obj.num_of_ts):
                nodes = list(range(net_obj.num_of_nodes))
                while len(nodes) > 0:
                    node = nodes.pop(0)
                    snd_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.from_], key=lambda c: c.to)
                    rcv_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.to], key=lambda c: c.from_)
                    # According to cp2modest.py all contacts must be full-duplex (reflexive) and only one full-duplex communication is allowed
                    if len(snd_contacts) == len(rcv_contacts):
                        if len(snd_contacts) > 0:
                            for i in random.sample(range(len(snd_contacts)), len(snd_contacts)):
                                if snd_contacts[i].to in nodes:
                                    contacts.append(snd_contacts[i])
                                    contacts.append(rcv_contacts[i])
                                    nodes.remove(snd_contacts[i].to)
                                    assert snd_contacts[i].to == rcv_contacts[i].from_, "must add a simetric contact"
                                    break
                    else:
                        assert False, f"node {node} has unidirectional contact in slot {ts}"

            sorted(contacts, key=lambda x: (x.ts, x.from_, x.to))
            net_obj = Net(net_obj.num_of_nodes, contacts)
            head, tail = os.path.split(path_to_net)
            net_obj.print_to_file(head, file_name=tail)
        else:
            net_obj = Net.get_net_from_file(path_to_net, contact_pf_required=False)

        net_obj.print_dtnsim_cp_to_file(ts_duration_in_seconds, 100, dtnsim_cp_path)