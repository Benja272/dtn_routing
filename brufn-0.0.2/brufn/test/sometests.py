import unittest
from brufn.network import *
import random
from brufn.brufspark import BRUFSpark
import itertools
import os
from pyspark import SparkContext, SparkConf
from brufn.net_metrics_generator import NetMetricGenerator
import simplejson as json
from copy import copy

class Testing(unittest.TestCase):

    '''
    AUXILIARY METHODS
    '''
    def gen_net1(self, num_of_nodes) -> Net:
        '''
        it generates a net in which i is linked with i+1. Besides, node i < num_of_nodes-2 has a direct contact to num_of_nodes-1
        :return:
        '''
        contacts = [Contact(x, x + 1, 0, identifier=x) for x in range(num_of_nodes - 1)]
        contacts += [Contact(x, num_of_nodes - 1, 0, identifier=x + len(contacts)) for x in range(num_of_nodes - 2)]

        return Net(num_of_nodes, contacts)

    # def one_sender_rules_aux_method(self, copies):
    #     net = self.gen_net1(6)
    #
    #     state = SoftState([0] * 5 + [copies], 0)
    #     bruf = BRUF(net, [0], 5, 3, [], '')
    #     self.assertEqual(state.num_of_carrying_copies(5), copies)
    #     hardcoded_routes = [
    #         [(0, 1, 2, 3, 4), (5,), (0, 6), (0, 1, 7), (0, 1, 2, 8)],  # from Node 0
    #         [(1, 2, 3, 4), (6,), (1, 7), (1, 2, 8)],  # from Node 1
    #         [(2, 3, 4), (7,), (2, 8)],  # from Node 2
    #         [(3, 4), (8,)],  # from Node 3
    #         [(4,)]  # from Node 4
    #     ]
    #
    #     required_rules = []
    #     for node in range(net.num_of_nodes - 1):
    #         hardcoded_rules_by_node = [(c, hardcoded_routes[node][i]) for i in range(len(hardcoded_routes[node])) for c in range(1, copies + 1)]
    #         for c in range(1, copies + 1):
    #             required_rules.append([])
    #             for rule in itertools.combinations(hardcoded_rules_by_node, c):
    #                 if sum([x[0] for x in rule]) <= copies:
    #                     required_rules[node].append(rule)
    #
    #     for sender in range(5):
    #         rules_copies_minus_1, rules = bruf.compute_one_sender_rules(sender, 5, 0, state.num_of_carrying_copies(5))
    #         # check number of rules
    #         self.assertEqual(len(required_rules[sender]), len(rules), 'Fail OneSenderRule with %d copies, for sender = %d'%(copies,sender))
    #         self.assertEqual(len(list(filter(lambda rule: sum([x[0] for x in rule]) < (state.num_of_carrying_copies(5)), required_rules[sender]))), rules_copies_minus_1)
    #         # check each rule
    #         for hr in required_rules[sender]:
    #             exist_rule = False
    #             for rule in rules:
    #                 if len(rule.route) == len(hr) and all(any(rule.route[i].contacts_ids == irh[1] and rule.copies[i] == irh[0]
    #                                                 for i in range(len(rule.route))) for irh in hr):
    #                     exist_rule = True
    #                     break
    #
    #             self.assertTrue(exist_rule,
    #                             "Fail OneSenderRule with %d copies from Sender %d: %s is not listed" % (
    #                             copies, sender, str(hr)))
    #
    #         for r in rules:
    #             self.assertTrue(isinstance(r, OneSenderRule), 'Rules must be instance of OneSenderRule')
    #             self.assertEqual(r.sender_node, [sender] * len(r.route), 'Sender node must be %d' % sender)
    #             self.assertEqual(r.receiver_node, [5] * len(r.route), 'Receiver node must be 5')

    # def many_sender_rules_aux_method(self, copies):
    #     net = self.gen_net1(6)
    #
    #     state = SoftState([0] * 5 + [copies], 0)
    #     bruf = BRUF(net, [0], 5, copies, [], '')
    #     self.assertEqual(state.num_of_carrying_copies(5), copies)
    #     hardcoded_routes = [
    #         [(0, 1, 2, 3, 4), (5,), (0, 6), (0, 1, 7), (0, 1, 2, 8)],  # from Node 0
    #         [(1, 2, 3, 4), (6,), (1, 7), (1, 2, 8)],  # from Node 1
    #         [(2, 3, 4), (7,), (2, 8)],  # from Node 2
    #         [(3, 4), (8,)],  # from Node 3
    #         [(4,)]  # from Node 4
    #     ]
    #
    #     hc_one_sender_rules_by_node = []
    #     for node in range(net.num_of_nodes - 1):
    #         hardcoded_rules_by_node = [(c, hardcoded_routes[node][i]) for i in range(len(hardcoded_routes[node])) for c
    #                                    in range(1, copies + 1)]
    #         for c in range(1, copies + 1):
    #             hc_one_sender_rules_by_node.append([])
    #             for rule in itertools.combinations(hardcoded_rules_by_node, c):
    #                 if sum([x[0] for x in rule]) <= copies:
    #                     hc_one_sender_rules_by_node[node].append(rule)
    #
    #     required_rules = []
    #     required_rules_senders = []
    #     generators = [(node, route) for node in range(net.num_of_nodes - 1) for route in
    #                   range(len(hc_one_sender_rules_by_node[node]) + 1)]
    #     for intended_rule in itertools.combinations(generators, net.num_of_nodes - 1):
    #         sender_nodes = [x[0] for x in intended_rule]
    #         sender_nodes_set = set(sender_nodes)
    #         if len(sender_nodes_set) > 1 and len(sender_nodes_set) == len(sender_nodes):  # Only allow one rule by node
    #             rule = [hc_one_sender_rules_by_node[r[0]][r[1] - 1] for r in intended_rule if r[1] > 0]
    #             if len(rule) > 1 and 0 < sum([r[0] for r in itertools.chain.from_iterable(rule)]) <= copies:
    #                 # Append only valid ManySenderRules. It means, more than 1 sender node which send proper number of copies
    #                 required_rules.append(list(itertools.chain.from_iterable(rule)))
    #                 required_rules_senders.append(set([x[0] for x in intended_rule if x[1] > 0]))
    #
    #     '''
    #     Generate rules using the test target code
    #     '''
    #     one_sender_rules = []
    #     for sender in range(net.num_of_nodes - 1):
    #         rules_copies_minus_1, rules = bruf.compute_one_sender_rules(sender, 5, 0, state.num_of_carrying_copies(5))
    #         one_sender_rules.append(rules[:rules_copies_minus_1])
    #
    #     one_sender_rules = [[x for x in node_rules] + [NextRule(-1, 0)] for node_rules in one_sender_rules]
    #
    #     many_sender_rules = bruf.compute_many_sender_rules(one_sender_rules, copies)
    #
    #     # Check number of rules
    #     self.assertEqual(len(required_rules), len(many_sender_rules), 'Fail ManySenderRules with %d copies' % (copies))
    #
    #     # Check that each required rule is listed
    #     for hr, sender_nodes in zip(required_rules, required_rules_senders):
    #         exist_rule = False
    #         for rule in many_sender_rules:
    #             if len(rule.route) == len(hr) and all(
    #                     any(rule.route[i].contacts_ids == irh[1] and rule.copies[i] == irh[0]
    #                         for i in range(len(rule.route))) for irh in hr):
    #                 exist_rule = True
    #                 self.assertTrue(isinstance(rule, ManySenderRule),
    #                                 'Rules must be instance of ManySenderRule. An instance of %s has been found' % type(
    #                                     rule))
    #                 self.assertTrue(all(x in rule.sender_node for x in sender_nodes), 'Sender node differs from expected: Expected %s in %s Rule: %s'%(sender_nodes, rule.sender_node, str(rule)))
    #                 self.assertEqual([5] * len(rule.route), rule.receiver_node, 'Receiver node must be 5. Rule: %s'%(str(rule)))
    #                 break
    #
    #         self.assertTrue(exist_rule, "Fail OneSenderRule with %d copies: %s is not listed"%(copies, str(hr)))

    '''
    TEST CASES
    '''

    def test_states_identifier(self):
        for copies in range(1, 5):
            counter = 0
            for ts in range(10):
                for i in range(copies + 1):
                    for j in range(copies + 1):
                        for k in range(copies + 1):
                            if i+j+k == copies:
                                s = SoftState((i,j,k), ts)
                                self.assertEqual(counter, s.id, "Should be %d"%counter)
                            counter += 1

    def test_getting_all_paths1(self):
        net = Net.get_net_from_file('c1.py', contact_pf_required=False)
        routes_0_1_0 = net.compute_routes(0, 1, 0)
        routes_0_2_0 = net.compute_routes(0, 2, 0)
        routes_1_2_0 = net.compute_routes(1, 2, 0)
        routes_1_2_1 = net.compute_routes(1, 2, 1)
        routes_0_2_2 = net.compute_routes(0, 2, 2)
        routes_1_2_2 = net.compute_routes(1, 2, 2)

        self.assertEqual(len(routes_0_1_0), 1, 'There must be 1 path from 0 to 1 at ts 0 but %d'%len(routes_0_1_0))
        self.assertEqual([x.id for x in routes_0_1_0[0].contacts], [0], 'send toward contact 0')

        self.assertEqual(len(routes_0_2_0), 1, 'There must be 1 path from 0 to 2 at ts 0 but %d'%len(routes_0_2_0))
        self.assertEqual([x.id for x in routes_0_2_0[0].contacts], [0,1], 'send toward contact 1')

        self.assertEqual(len(routes_1_2_0), 1, 'There must be 1 path from 1 to 2 at ts 0 but %d'%len(routes_1_2_0))
        self.assertEqual([x.id for x in routes_1_2_0[0].contacts], [1], 'send toward contact 1')

        self.assertEqual(len(routes_1_2_1), 1, 'There must be 1 path from 1 to 2 at ts 1 but %d'%len(routes_1_2_1))
        self.assertEqual([x.id for x in routes_1_2_1[0].contacts], [2], 'send toward contact 1')

        self.assertEqual(len(routes_0_2_2), 1, 'There must be 1 path from 0 to 2 at ts 2 but %d'%len(routes_0_2_2))
        self.assertEqual([x.id for x in routes_0_2_2[0].contacts], [3], 'send toward contact 1')

        self.assertEqual(len(routes_1_2_2), 1, 'There must be 1 path from 1 to 2 at ts 2 but %d'%len(routes_1_2_2))
        self.assertEqual([x.id for x in routes_1_2_2[0].contacts], [4], 'send toward contact 1')

    def test_getting_all_paths2(self):
        net = self.gen_net1(6)
        routes_0_5_0 = net.compute_routes(0, net.num_of_nodes - 1, 0)

        self.assertEqual(len(routes_0_5_0), 5, 'There must be 5 path from 0 to 5 at ts 0 but %d' % len(routes_0_5_0))

        hardcoded_routes = [[0,1,2,3,4], [5], [0, 6], [0,1,7], [0,1,2,8]]
        computed_routes = [[x.id for x in r.contacts] for r in routes_0_5_0]

        self.assertTrue(all(x in computed_routes for x in hardcoded_routes), 'Expected %s get %s'%(hardcoded_routes, computed_routes))

    def test_getting_all_paths3(self):
        net = self.gen_net1(6)
        routes_1_5_0 = net.compute_routes(1, net.num_of_nodes - 1, 0)

        self.assertEqual(len(routes_1_5_0), 4, 'There must be 5 path from 0 to 5 at ts 0 but %d' % len(routes_1_5_0))

        hardcoded_routes = [[1, 2, 3, 4], [6], [1, 7], [1,2,8]]
        computed_routes = [[x.id for x in r.contacts] for r in routes_1_5_0]

        self.assertTrue(all(x in computed_routes for x in hardcoded_routes), 'Expected %s get %s'%(hardcoded_routes, computed_routes))

    def test_getting_all_paths_nonloop(self):
        contacts = [Contact(0, 1, 0, identifier=0), Contact(1, 2, 0, identifier=1), Contact(1, 0, 0, identifier=2), Contact(0, 2, 0, identifier=3)]

        net = Net(3, contacts)

        routes_1_5_0 = net.compute_routes(0, 2, 0)

        self.assertEqual(len(routes_1_5_0), 2, 'There must be 5 path from 0 to 5 at ts 0 but %d' % len(routes_1_5_0))

        hardcoded_routes = [[3],[0,1]]
        computed_routes = [[x.id for x in r.contacts] for r in routes_1_5_0]

        self.assertTrue(all(x in computed_routes for x in hardcoded_routes), 'Expected %s get %s'%(hardcoded_routes, computed_routes))

    def test_get_carrying_nodes(self):
        for i in range(100):
            states = [0 if x < 0.5 else random.randint(1, 4) for x in [random.random() for i in range(5)]]
            s = SoftState(states, 0)
            self.assertEqual(s.get_carrier_nodes(), [i for i in range(5) if states[i] > 0], "")

    def test_listfailures1(self):
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)

        net = Net(2, [Contact(0,1, 1)]) # Dummy net
        bruf = BRUFSpark(net, [0], 1, 1, [], 'working_dir') #Dummy BRUFME

        to_state = SoftState([0,1], 2) #Dummy to_state
        rules = [Rule(1, Route([net.contacts[0]]))] #Dummy rule
        computed_from_ = to_state.gen_previous_state(rules)

        expected_success_from = SoftState([1, 0], 1)
        self.assertEqual(expected_success_from.states, computed_from_) # Check expected from

        t = Transition(SoftState(computed_from_,0), to_state, rules)
        t.compute_changes(sc.broadcast({to_state.id: to_state, SoftState([1, 0], 2).id:SoftState([1, 0], 2)}))
        sc.stop()

        #Hardcode expected changes to validate the computed ones
        expected_change_states_list = sorted([to_state.id, SoftState([1, 0], 2).id])
        expected_change_case_list = ((1,), (0,))

        self.assertEqual(2, len(t._changes_list))
        sorted_change_list = sorted(t._changes_list)
        for i in range(len(sorted_change_list)):
            c:Change = t._changes_list[sorted_change_list[i]]

            self.assertEqual(c.to_state.id, expected_change_states_list[i])
            self.assertEqual(1, len(c._cases))
            cc: ChangeCase = c._cases[0]
            self.assertEqual([net.contacts[0].id], [x.id for x in cc._contacts])
            self.assertEqual(expected_change_case_list[i], cc._case)

    def test_listfailures2(self):
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)

        net = Net(2, [Contact(0,1, 1)]) # Dummy net
        bruf = BRUFSpark(net, [0], 1, 1, [], 'working_dir') #Dummy BRUFME

        to_state = SoftState([0,1], 2) #Dummy to_state
        rules = [Rule(1, Route([net.contacts[0]]))] #Dummy rule
        computed_from_ = to_state.gen_previous_state(rules)

        expected_success_from = SoftState([1, 0], 1)
        self.assertEqual(expected_success_from.states, computed_from_) # Check expected from

        t = Transition(SoftState(computed_from_,0), to_state, rules)
        t.compute_changes(sc.broadcast({to_state.id: to_state}))
        sc.stop()

        #Hardcode expected changes to validate the computed ones
        expected_change_states_list = sorted([to_state.id])
        expected_change_case_list = ((1,),)

        self.assertEqual(1, len(t._changes_list))
        sorted_change_list = sorted(t._changes_list)
        for i in range(len(sorted_change_list)):
            c:Change = t._changes_list[sorted_change_list[i]]

            self.assertEqual(c.to_state.id, expected_change_states_list[i])
            self.assertEqual(1, len(c._cases))
            cc: ChangeCase = c._cases[0]
            self.assertEqual([net.contacts[0].id], [x.id for x in cc._contacts])
            self.assertEqual(expected_change_case_list[i], cc._case)


    def test_check_builded_tree(self):
        pass

    def test_pr1(self):
        pass

    def test_pr2(self):
        pass

    def test_final_states(self):
        net = Net(8, [Contact(0,7,0)]) # empy net

        for copies in range(1, 5):
            for t in range(7):
                bruf = BRUFSpark(net, [0], 3, copies, [], 'working_dir')
                required_final_states = sorted([SoftState.get_identifier(x, 1)
                                        for x in itertools.product(range(copies + 1), repeat=8) if sum(x) == copies and x[t] > 0])
                computed_final_states = []
                for c in range(1, copies + 1):
                    computed_final_states.append([SoftState.get_identifier(s, 1) for s in bruf.gen_final_states(copies - c, t, c)])
                computed_final_states = sorted(list(itertools.chain.from_iterable(computed_final_states)))

                self.assertEqual(required_final_states, computed_final_states, 'FinalState fail For %d copies, target %d'%(copies, t))


    def test_get_new_state_by_identifier(self):
        num_of_copies = 3; nodes_number = 3; ts_number = 20;
        for ts in range(ts_number):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        expected_id = SoftState.get_identifier((i,j,k), ts)
                        if i+j+k == num_of_copies:
                            s = SoftState.get_new_state_by_identifier(expected_id, ts_number, nodes_number, num_of_copies)
                            self.assertEqual(s.id, expected_id)
                            self.assertEqual(s.states, (i,j,k))
                            self.assertEqual(s.ts, ts)

        #expected_id = SoftState.get_identifier((num_of_copies + 1, num_of_copies + 1, num_of_copies + 1), ts_number - 1)
        #self.assertRaises(ValueError, SoftState.get_new_state_by_identifier, expected_id, ts_number, nodes_number, num_of_copies)

    def test_generate_statictiscs_1(self):
        contacts = [Contact(0, 1, 0, begin_time=0, end_time=1)]
        net = Net(5, contacts)
        statistics = net.generate_statictiscs()
        self.assertEqual(statistics['distance_contact_end_to_ts_end-avg'], 0)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-min'], 0)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-max'], 0)

        self.assertEqual(statistics['distance_contact_start_to_ts_start-avg'], 0)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-min'], 0)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-max'], 0)

        self.assertEqual(statistics['ts_duration-avg'], 1)
        self.assertEqual(statistics['ts_duration-min'], 1)
        self.assertEqual(statistics['ts_duration-max'], 1)

        self.assertEqual(statistics['ts_number_of_contacts-avg'], 1)
        self.assertEqual(statistics['ts_number_of_contacts-min'], 1)
        self.assertEqual(statistics['ts_number_of_contacts-max'], 1)

    def test_generate_statictiscs_2(self):
        contacts = [
                    Contact(0, 1, 0, begin_time=0, end_time=5),
                    Contact(1, 2, 0, begin_time=1, end_time=4),
                    Contact(2, 3, 0, begin_time=2, end_time=3),

                    Contact(0, 1, 1, begin_time=5, end_time=10),
                    Contact(1, 2, 1, begin_time=6, end_time=9),

                    Contact(0, 1, 2, begin_time=10, end_time=14),
                    Contact(1, 2, 2, begin_time=10, end_time=15),
        ]

        net = Net(5, contacts)
        statistics = net.generate_statictiscs()
        self.assertAlmostEqual(statistics['distance_contact_end_to_ts_end-avg'], 5/7)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-min'], 0)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-max'], 2)

        self.assertAlmostEqual(statistics['distance_contact_start_to_ts_start-avg'], 4/7)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-min'], 0)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-max'], 2)

        self.assertEqual(statistics['ts_duration-avg'], 5)
        self.assertEqual(statistics['ts_duration-min'], 5)
        self.assertEqual(statistics['ts_duration-max'], 5)

        self.assertAlmostEqual(statistics['ts_number_of_contacts-avg'], 7/3.)
        self.assertEqual(statistics['ts_number_of_contacts-min'], 2)
        self.assertEqual(statistics['ts_number_of_contacts-max'], 3)



    def test_generate_statictiscs_3(self):
        contacts = [
                    Contact(0, 1, 0, begin_time=0, end_time=5),
                    Contact(1, 2, 0, begin_time=1, end_time=4),
                    Contact(2, 3, 0, begin_time=2, end_time=3),

                    Contact(0, 1, 1, begin_time=5, end_time=10),
                    Contact(1, 2, 1, begin_time=6, end_time=9),

                    Contact(0, 1, 2, begin_time=10, end_time=14),
                    Contact(1, 2, 2, begin_time=10, end_time=15),
                    Contact(2, 3, 2, begin_time=14, end_time=20),
        ]

        net = Net(5, contacts)
        statistics = net.generate_statictiscs()
        self.assertAlmostEqual(statistics['distance_contact_end_to_ts_end-avg'], (0+1+2+0+1+6+5+0)/8)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-min'], 0)
        self.assertEqual(statistics['distance_contact_end_to_ts_end-max'], 6)

        self.assertAlmostEqual(statistics['distance_contact_start_to_ts_start-avg'], (0+1+2+0+1+0+0+4)/8)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-min'], 0)
        self.assertEqual(statistics['distance_contact_start_to_ts_start-max'], 4)

        self.assertEqual(statistics['ts_duration-avg'], 20/3)
        self.assertEqual(statistics['ts_duration-min'], 5)
        self.assertEqual(statistics['ts_duration-max'], 10)

        self.assertAlmostEqual(statistics['ts_number_of_contacts-avg'], 8/3.)
        self.assertEqual(statistics['ts_number_of_contacts-min'], 2)
        self.assertEqual(statistics['ts_number_of_contacts-max'], 3)

    def test_contact_equal_method(self):
        c = Contact(0, 1, 1)
        self.assertEqual(c, copy(c))
        self.assertNotEqual(c, Contact(1,0,1))
        self.assertNotEqual(c, Contact(0, 2, 1))
        self.assertNotEqual(c, Contact(0, 1, 2))
        self.assertNotEqual(c, Contact(0, 1, 1, pf=0.1))
        self.assertNotEqual(c, Contact(0, 1, 1, begin_time=10))
        self.assertNotEqual(c, Contact(0, 1, 1, end_time=10))

        c = Contact(0, 1, 1, pf=0.1, begin_time=10, end_time=20)
        self.assertEqual(c, copy(c))
        self.assertNotEqual(c, Contact(0, 1, 1))
        self.assertNotEqual(c, Contact(0, 1, 1, pf=0.2, begin_time=10, end_time=20))
        self.assertNotEqual(c, Contact(0, 1, 1, pf=0.1, begin_time=20, end_time=20))
        self.assertNotEqual(c, Contact(0, 1, 1, pf=0.1, begin_time=10, end_time=10))
        self.assertEqual(c, Contact(0, 1, 1, pf=0.1, begin_time=10, end_time=20))


    def test_contact_equal_and_hash_method(self):
        c = Contact(0, 1, 1)
        self.assertEqual(len(set([c,copy(c)])), 1)
        self.assertEqual(len(set([c, copy(c), Contact(0,1,2), Contact(1,0,1), Contact(1,2,1)])), 4)
        self.assertEqual(len(set([c, Contact(0,1,2), Contact(1,0,1), Contact(1,2,1)])), 4)
        self.assertEqual(len(set([Contact(2, 1, 1), Contact(1, 0, 1), Contact(1, 2, 1), Contact(1, 0, 1)])), 3)


class TestIntegration(unittest.TestCase):

    def setUp(self):
        os.system('mkdir working_dir')
        os.system('mkdir working_dir_1')
        os.system('mkdir working_dir_2')

    def tearDown(self):
        os.system('rm -r working_dir')
        os.system('rm -r working_dir_1')
        os.system('rm -r working_dir_2')

    def test_genfinalprobabilities_1(self):
        net = Net.get_net_from_file('twopath.py')
        bruf = BRUFSpark(net, [0], 3, 2, [], 'working_dir')
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)
        bruf.compute_bruf(sc)
        sc.stop()

        states = [((2,0,0,0), 0) , ((0,0,0,2), 1), ((0,0,1,1), 1), ((0,1,0,1), 1), ((1,0,0,1),1)]
        max_success_pr = [0.9*0.8 + 0.85 * 0.7 - 0.9*0.8 * 0.85 * 0.7, 1., 1., 1., 1. ]
        max_success_rule = [ [(1, (1,3)), (1, (2,4))], None, None, None, None]

        self.assertEqual(len(states), bruf.states_number)
        for s, pr, msrule in zip(states, max_success_pr, max_success_rule):
            save_s = bruf.get_state_by_id(SoftState.get_identifier(s[0], s[1]))
            self.assertEqual(s, (save_s['states'], save_s['ts']) )
            self.assertAlmostEqual(pr, save_s[f'sdp_pf=-1'], delta=0.0000001)
            if msrule is None:
                self.assertEqual(msrule, save_s['best_t_pf=-1'])
            else:
                t = self.to_tuple(save_s['best_t_pf=-1'])
                self.assertTrue(all(r in t for r in msrule) and len(msrule) == len(t), "Expected %s Get %s"%(msrule, t))
                to_states = save_s['t_changes_pf=-1']
                for x in [s for s in states if s[1]==1]:
                    self.assertTrue(SoftState.get_identifier(x[0], x[1]) in to_states)

    def test_genfinalprobabilities_2(self):
        net = Net.get_net_from_file('twopath_twots.py')
        bruf = BRUFSpark(net, [0], 3, 2, [], 'working_dir')
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)
        bruf.compute_bruf(sc)
        sc.stop()

        states = [((2,0,0,0), 0), ((0,0,2,0), 1), ((0,2,0,0), 1), ((0,1,1,0), 1), ((1,1,0,0), 1), ((1,0,1,0), 1), ((0,0,0,2), 2), ((0,0,1,1), 2), ((0,1,0,1), 2), ((1,0,0,1),2)]
        max_success_pr = [0.9 * 0.8 + 0.85 * 0.7 - 0.9 * 0.8 * 0.85 * 0.7, 0.7, 0.8, 1 - (0.2 * 0.3), 0.8, 0.7, 1., 1., 1., 1.]
        max_success_rule = [[(1, (1,)), (1, (2,))], [(2,(4,))], [(2,(3,))], [(1, (3,)), (1, (4,))], [(1,(3,))], [(1,(4,))], None, None, None, None]
        states_not_reacheables_from_root = [((0, 0, 0, 2), 1), ((0, 1, 0, 1), 1), ((0, 0, 1, 1), 1), ((1, 0, 0, 1), 1)]
        states_ids = [SoftState.get_identifier(x[0], x[1]) for x in states] + [SoftState.get_identifier(x[0], x[1]) for x in states_not_reacheables_from_root]
        assert len(states) == len(max_success_pr) == len(max_success_rule), 'Error in test case formulation'
        self.assertEqual(len(states) + len(states_not_reacheables_from_root), bruf.states_number)

        for s, pr, msrule in zip(states, max_success_pr, max_success_rule):
            save_s = bruf.get_state_by_id(SoftState.get_identifier(s[0], s[1]))
            self.assertEqual(s, (save_s['states'], save_s['ts']))
            self.assertAlmostEqual(pr, save_s[f'sdp_pf=-1'], delta=0.0000001)
            if msrule is None:
                self.assertEqual(msrule, save_s['best_t_pf=-1'])
            else:
                t = self.to_tuple(save_s['best_t_pf=-1'])
                self.assertTrue(all(r in t for r in msrule) and len(msrule) == len(t), "Expected %s Get %s"%(msrule, t))
                to_states = save_s['t_changes_pf=-1']
                for x in to_states:
                    self.assertTrue(x in states_ids)

    def test_genfinalprobabilities_rng_1(self):
        net = Net.get_net_from_file('twopath.py')
        pf_rng = [x/100 for x in range(0, 110, 10)]
        bruf = BRUFSpark(net, [0], 3, 2, pf_rng, 'working_dir')
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)
        bruf.compute_bruf(sc)
        sc.stop()

        for pf in pf_rng:
            if pf == 0 or pf == 1:
                max_success_rule = [[(2, (2, 4))], None, None, None, None]
            else:
                max_success_rule = [ [(1, (1,3)), (1, (2,4))], None, None, None, None]
            max_success_pr = [(1-pf) * (1-pf) + (1-pf) * (1-pf) - (1-pf) ** 4, 1., 1., 1., 1. ]
            states = [((2, 0, 0, 0), 0), ((0, 0, 0, 2), 1), ((0, 0, 1, 1), 1), ((0, 1, 0, 1), 1), ((1, 0, 0, 1), 1)]

            self.assertEqual(len(states), bruf.states_number)
            for s, pr, msrule in zip(states, max_success_pr, max_success_rule):
                save_s = bruf.get_state_by_id(SoftState.get_identifier(s[0], s[1]))
                self.assertEqual(s, (save_s['states'], save_s['ts']))
                self.assertAlmostEqual(pr, save_s[f'sdp_pf={pf}'], delta=0.0000001)
                if msrule is None:
                   self.assertEqual(msrule,  save_s[f'best_t_pf={pf:.2}'])
                else:
                    t = self.to_tuple(save_s[f'best_t_pf={pf:.2}'])
                    self.assertTrue(all(r in t for r in msrule) and len(msrule) == len(t), "Expected %s Get %s"%(msrule, t)) #It fails because two trasition are equally good and it depens on the order of an iteration
                    to_states = save_s[f't_changes_pf={pf:.2}']
                    if pf == 0 or pf == 1:
                        self.assertTrue(SoftState.get_identifier([0,0,0,2], 1) in to_states)
                    else:
                        for x in [s for s in states if s[1]==1]:
                            self.assertTrue(SoftState.get_identifier(x[0], x[1]) in to_states)

    def test_genfinalprobabilities_rng_rc_1(self):
        net = Net.get_net_from_file('twopath.py')
        NetMetricGenerator(net, range(net.num_of_nodes), [2], [0], 'working_dir')
        pf_rng = [0., 0.5, 1.] #[x/100 for x in range(0, 110, 10)]
        bruf = BRUFSpark(net, [0], 3, 2, pf_rng, 'working_dir')
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)
        rc = json.load(open('working_dir/transitive_closure.json'), object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in d.items()})
        bruf.compute_bruf(sc, num_states_per_slice=10, reachability_closure=rc)
        sc.stop()

        for pf in pf_rng:
            if pf == 0 or pf == 1:
                max_success_rule = [[(2, (2, 4))], None, None, None, None]
            else:
                max_success_rule = [ [(1, (1,3)), (1, (2,4))], None, None, None, None]
            max_success_pr = [(1-pf) * (1-pf) + (1-pf) * (1-pf) - (1-pf) ** 4, 1., 1., 1., 1. ]
            states = [((2, 0, 0, 0), 0), ((0, 0, 0, 2), 1), ((0, 0, 1, 1), 1), ((0, 1, 0, 1), 1), ((1, 0, 0, 1), 1)]

            self.assertEqual(len(states), bruf.states_number)
            for s, pr, msrule in zip(states, max_success_pr, max_success_rule):
                save_s = bruf.get_state_by_id(SoftState.get_identifier(s[0], s[1]))
                self.assertEqual(s, (save_s['states'], save_s['ts']))
                self.assertAlmostEqual(pr, save_s[f'sdp_pf={pf}'], delta=0.0000001)
                if msrule is None:
                   self.assertEqual(msrule,  save_s[f'best_t_pf={pf:.2}'])
                else:
                    t = self.to_tuple(save_s[f'best_t_pf={pf:.2}'])
                    self.assertTrue(all(r in t for r in msrule) and len(msrule) == len(t), "Expected %s Get %s"%(msrule, t)) #It fails because two trasition are equally good and it depens on the order of an iteration
                    to_states = save_s[f't_changes_pf={pf:.2}']
                    if pf == 0 or pf == 1:
                        self.assertTrue(SoftState.get_identifier([0,0,0,2], 1) in to_states)
                    else:
                        for x in [s for s in states if s[1]==1]:
                            self.assertTrue(SoftState.get_identifier(x[0], x[1]) in to_states)


    def test_genfinalprobabilities_rng_rc_2(self):
        for net_id in range(1):
            net = Net.get_net_from_dtnsim_cp(f'10NetICC/0.2_{net_id}', 10)
            pf_rng = [0., 0.5,  1.0] #[x/100 for x in range(0, 110, 10)]
            sources = range(8)
            for copies in range(1, 3):
                for target in range(0, 1):
                    bruf = BRUFSpark(net, sources, target, copies, pf_rng, 'working_dir')
                    conf = SparkConf().setAppName("BRUF-Spark")
                    conf = (conf.setMaster('local[2]')
                            .set('spark.executor.memory', '2G')
                            .set('spark.driver.memory', '4G')
                            .set('spark.driver.maxResultSize', '8G'))
                    sc = SparkContext(conf=conf)
                    bruf.compute_bruf(sc, num_states_per_slice=10)
                    sc.stop()

                    bruf_rc = BRUFSpark(net, sources, target, copies, pf_rng, 'working_dir_1')
                    NetMetricGenerator(net, range(net.num_of_nodes), [copies], range(net.num_of_nodes), 'working_dir_1')
                    conf = SparkConf().setAppName("BRUF-Spark")
                    conf = (conf.setMaster('local[2]')
                            .set('spark.executor.memory', '2G')
                            .set('spark.driver.memory', '4G')
                            .set('spark.driver.maxResultSize', '8G'))
                    sc = SparkContext(conf=conf)
                    rc = json.load(open('working_dir_1/transitive_closure.json'), object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in d.items()})
                    bruf_rc.compute_bruf(sc, num_states_per_slice=10, reachability_closure=rc)
                    sc.stop()

                    for pf in pf_rng:
                        for source in sources:
                            save_s = bruf.get_state_by_id(SoftState.get_identifier([0 if x != source else copies for x in range(net.num_of_nodes)],0))
                            save_s_rc = bruf_rc.get_state_by_id(SoftState.get_identifier([0 if x != source else copies for x in range(net.num_of_nodes)], 0))
                            self.assertAlmostEqual(save_s[f'sdp_pf={pf}'], save_s_rc[f'sdp_pf={pf}'], delta=0.0000001)

    def test_genfinalprobabilities_pr_vs_rng(self):
        # It compares results computed which range version and without range
        pf_rng = [0, 0.5, 1.] #[x / 100 for x in range(0, 110, 10)]
        sources = range(8)

        for net_id in range(1):
            bruf = {}
            net = Net.get_net_from_dtnsim_cp(f'10NetICC/0.2_{net_id}', 10)
            NetMetricGenerator(net, range(net.num_of_nodes), [1], range(net.num_of_nodes), 'working_dir_2')
            for copies in range(1, 3):
                for target in range(7, 8):
                    bruf_pr_rng = BRUFSpark(net, sources, target, copies, pf_rng, 'working_dir_1')
                    conf = SparkConf().setAppName("BRUF-Spark")
                    conf = (conf.setMaster('local[2]')
                            .set('spark.executor.memory', '2G')
                            .set('spark.driver.memory', '4G')
                            .set('spark.driver.maxResultSize', '8G'))
                    sc = SparkContext(conf=conf)
                    bruf_pr_rng.compute_bruf(sc, num_states_per_slice=10)
                    sc.stop()
                    bruf_pr_rng.generate_mc_to_dtnsim_all_sources_all_pf('working_dir_1')

                    bruf_pr_rng_rc = BRUFSpark(net, sources, target, copies, pf_rng, 'working_dir_2')
                    conf = SparkConf().setAppName("BRUF-Spark")
                    conf = (conf.setMaster('local[2]')
                            .set('spark.executor.memory', '2G')
                            .set('spark.driver.memory', '4G')
                            .set('spark.driver.maxResultSize', '8G'))
                    sc = SparkContext(conf=conf)
                    rc = json.load(open('working_dir_2/transitive_closure.json'),
                                   object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v
                                                          in d.items()})
                    bruf_pr_rng_rc.compute_bruf(sc, num_states_per_slice=10, reachability_closure=rc)
                    sc.stop()
                    bruf_pr_rng_rc.generate_mc_to_dtnsim_all_sources_all_pf('working_dir_2')

                    for pf in pf_rng:
                        for c in net.contacts:
                            c.pf = pf
                        bruf[str(pf)] = BRUFSpark(net, sources, target, copies, [], 'working_dir')
                        conf = SparkConf().setAppName("BRUF-Spark")
                        conf = (conf.setMaster('local[2]')
                                .set('spark.executor.memory', '2G')
                                .set('spark.driver.memory', '4G')
                                .set('spark.driver.maxResultSize', '8G'))
                        sc = SparkContext(conf=conf)
                        bruf[str(pf)].compute_bruf(sc, num_states_per_slice=10)
                        sc.stop()
                        bruf[str(pf)].generate_mc_to_dtnsim_all_sources_all_pf('working_dir')

                        for source in sources:
                            save_s = bruf[str(pf)].get_state_by_id(SoftState.get_identifier([0 if x != source else copies for x in range(net.num_of_nodes)], 0))
                            save_s_pr_rng = bruf_pr_rng.get_state_by_id(SoftState.get_identifier([0 if x != source else copies for x in range(net.num_of_nodes)], 0))
                            self.assertAlmostEqual(save_s[f'sdp_pf=-1'], save_s_pr_rng[f'sdp_pf={pf}'], delta=0.0000001)
                            with open(f'working_dir/pf=-1.00/todtnsim-{source}-{target}--1.00.json','r')  as f_bruf,\
                                 open(f'working_dir_1/pf={pf:.2f}/todtnsim-{source}-{target}-{pf:.2f}.json','r') as f_bruf_pr_rng,\
                                 open(f'working_dir_2/pf={pf:.2f}/todtnsim-{source}-{target}-{pf:.2f}.json','r') as f_bruf_pr_rng_rc:
                                f_bruf_lines = f_bruf.readlines()
                                self.assertEqual(f_bruf_pr_rng.readlines(), f_bruf_lines)
                                self.assertEqual(f_bruf_pr_rng_rc.readlines(), f_bruf_lines)

    def test_bounded_iterator_1(self):
        expected_res = [[(0,1)], [(1,1)], [(2,1)], [(3,1)],[(4,1)],[(5,1)]]
        res = [x for x in bounded_iterator(6, 1)]

        self.assertEqual(len(expected_res), len(res))
        self.assertTrue(all(x in res for x in expected_res))

    def test_bounded_iterator_2(self):
        expected_res = [ [(0,2)], [(1,2)], [(2,2)],
                         [(0,1), (1,1)], [(0,1), (2,1)],
                         [(1,1), (2,1)]
                        ]
        res = [x for x in bounded_iterator(3, 2)]

    def test_bounded_iterator_3(self):
        expected_res = [[(0, 3)], [(1, 3)], [(2, 3)],
                        [(0, 2), (1, 1)], [(0, 2), (2, 1)],
                        [(1, 2), (2, 1)],
                        [(0, 1), (1,2)], [(0,1), (2,2)],
                        [(1,1), (2,2)],
                        [(0, 1), (1, 1), (2,1)]
                        ]
        res = [x for x in bounded_iterator(3, 3)]

        self.assertEqual(len(expected_res), len(res))
        self.assertTrue(all(x in res for x in expected_res))

    def test_cartesian_product(self):
        tuples_3_3 = [[(0, 3)], [(1, 3)], [(2, 3)],
                        [(0, 2), (1, 1)], [(0, 2), (2, 1)],
                        [(1, 2), (2, 1)],
                        [(0, 1), (1,2)], [(0,1), (2,2)],
                        [(1,1), (2,2)],
                        [(0, 1), (1, 1), (2,1)]
                        ]
        tuples_3_2 = [[(0,2)], [(1,2)], [(2,2)],
                         [(0,1), (1,1)], [(0,1), (2,1)],
                         [(1,1), (2,1)]
                      ]
        res = [bounded_iterator(3, 3), bounded_iterator(3, 2)]
        cartesian_product = [x for x in itertools.product(*res)]
        self.assertEqual(len(tuples_3_3) * len(tuples_3_2), len(cartesian_product))
        for x in tuples_3_3:
            for y in tuples_3_2:
                self.assertTrue((x, y) in cartesian_product)

    '''
    Auxiliary methods
    '''

    def to_tuple(self, list_dict_rule) -> Tuple[int, Tuple[int]]:
        res = []
        for dict_rule in list_dict_rule:
            if 'next' not in dict_rule['name']:
                res.append((dict_rule['copies'], tuple(dict_rule['contact_ids'])))

        return res



if __name__ == '__main__':
    unittest.main()
