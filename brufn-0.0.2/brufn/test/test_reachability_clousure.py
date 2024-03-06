import unittest
from brufn.net_reachability_closure import reachability_clousure
from brufn.network import Net, Contact


def gen_binomial_net(levels):
    assert levels > 0, "Levels must be greater than 1"
    contacts = []
    for level in range(1, levels - 1):
        target = int(2 ** level - 1)
        for node in range(int(2 ** (level - 1) - 1), int(2 ** level - 1)):
            contacts.append(Contact(node, target, 0))
            target += 1
            contacts.append(Contact(node, target, 0))
            target += 1

    # Link nodes levels - 2 to target node
    for node in range(int(2 ** (levels - 2) - 1), int(2 ** (levels - 1) - 1)):
        contacts.append(Contact(node, 2 ** (levels - 1) - 1, 0))

    return Net(2 ** (levels - 1), contacts)

def gen_binomial_net_level_by_ts(levels):
    assert levels > 0, "Levels must be greater than 1"
    contacts = []
    for level in range(1, levels - 1):
        target = int(2 ** level - 1)
        for node in range(int(2 ** (level - 1) - 1), int(2 ** level - 1)):
            contacts.append(Contact(node, target, level - 1))
            target += 1
            contacts.append(Contact(node, target, level - 1))
            target += 1

    # Link nodes levels - 2 to target node
    for node in range(int(2 ** (levels - 2) - 1), int(2 ** (levels - 1) - 1)):
        contacts.append(Contact(node, 2 ** (levels - 1) - 1, levels - 2))

    return Net(2 ** (levels - 1), contacts)

class Test_Reachability_Clousure(unittest.TestCase):
    S = 0; A = 1; B = 2; C = 3; E = 4; D = 5;
    working_dir = 'output'

    def test_transitive_closure_1(self):
        net = Net.get_net_from_file("/home/fraverta/development/BRUF-WithCopies19/brufn/test/tests_metrics_generator/nets/net.py",
                                    contact_pf_required=False)
        expected_rc = {
                        self.S: {0: [self.S, self.A, self.B, self.C, self.D, self.E], 1: [self.S], 2: [self.S]},
                        self.A: {0: [self.A, self.B, self.D], 1: [self.A], 2: [self.A]},
                        self.B: {0: [self.B, self.D], 1: [self.B, self.D], 2: [self.B]},
                        self.C: {0: [self.C, self.D, self.E], 1: [self.C, self.D, self.E], 2: [self.C, self.D, self.E]},
                        self.D: {0: [self.D], 1: [self.D], 2: [self.D]},
                        self.E: {0: [self.E, self.D], 1: [self.E, self.D], 2: [self.E, self.D]},

        }

        rc = reachability_clousure(net)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))


    def test_transitive_closure_2(self):
        net = gen_binomial_net_level_by_ts(4)
        expected_rc = {
                        0: {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [0], 2: [0]},
                        1: {0: [1, 3, 4, 7], 1: [1, 3, 4, 7], 2: [1]},
                        2: {0: [2, 5, 6, 7], 1: [2, 5, 6, 7], 2: [2]},
                        3: {0: [3, 7], 1: [3, 7], 2: [3, 7]},
                        4: {0: [4, 7], 1: [4, 7], 2: [4, 7]},
                        5: {0: [5, 7], 1: [5, 7], 2: [5, 7]},
                        6: {0: [6, 7], 1: [6, 7], 2: [6, 7]},
                        7: {0: [7], 1: [7], 2: [7]},

        }

        rc = reachability_clousure(net)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))



    def test_transitive_closure_3(self):
        net = gen_binomial_net(4)
        expected_rc = {
                        0: {0: [0, 1, 2, 3, 4, 5, 6, 7]},
                        1: {0: [1, 3, 4, 7]},
                        2: {0: [2, 5, 6, 7]},
                        3: {0: [3, 7]},
                        4: {0: [4, 7]},
                        5: {0: [5, 7]},
                        6: {0: [6, 7]},
                        7: {0: [7]},

        }

        rc = reachability_clousure(net)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))


    def test_transitive_closure_4(self):
        contacts = [Contact(0, 1, 0), Contact(1, 0, 0),
                    Contact(1, 2, 1), Contact(2, 1, 1),
                    Contact(2, 3, 2), Contact(3, 2, 2),
                    ]
        net = Net(4, contacts)
        expected_rc = {
                        0: {0: [0, 1, 2, 3], 1: [0],       2: [0]},
                        1: {0: [0, 1, 2, 3], 1: [1, 2, 3], 2: [1]},
                        2: {0: [1, 2, 3],    1: [1, 2, 3], 2: [2, 3]},
                        3: {0: [2, 3],       1: [2, 3],    2: [2, 3]},
        }

        rc = reachability_clousure(net)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))


    def test_transitive_closure_from_source_1(self):
        net = Net.get_net_from_file("/home/fraverta/development/BRUF-WithCopies19/brufn/test/tests_metrics_generator/nets/net.py",
                                    contact_pf_required=False)
        expected_rc = {
                        -1:[self.S],
                        0: [self.S, self.A, self.B, self.C],
                        1: [self.S, self.A, self.B, self.C, self.D],
                        2: [self.S, self.A, self.B, self.C, self.D, self.E],
        }

        rc = net.reachability_clousure_from_source(0)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for ts in range(net.num_of_ts):
            self.assertEqual(len(expected_rc[ts]), len(rc[ts]))
            self.assertTrue(all(n in rc[ts] for n in expected_rc[ts]))


    def test_transitive_closure_max_hop_dummy(self):
        net = gen_binomial_net_level_by_ts(4)
        expected_rc = {
                        0: {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [0], 2: [0]},
                        1: {0: [1, 3, 4, 7], 1: [1, 3, 4, 7], 2: [1]},
                        2: {0: [2, 5, 6, 7], 1: [2, 5, 6, 7], 2: [2]},
                        3: {0: [3, 7], 1: [3, 7], 2: [3, 7]},
                        4: {0: [4, 7], 1: [4, 7], 2: [4, 7]},
                        5: {0: [5, 7], 1: [5, 7], 2: [5, 7]},
                        6: {0: [6, 7], 1: [6, 7], 2: [6, 7]},
                        7: {0: [7], 1: [7], 2: [7]},

        }

        rc = reachability_clousure(net, max_hops_route_in_ts=10)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))

    def test_transitive_closure_max_hop_dummy_2(self):
        net = gen_binomial_net_level_by_ts(4)
        expected_rc = {
                        0: {0: [0], 1: [0], 2: [0]},
                        1: {0: [1], 1: [1], 2: [1]},
                        2: {0: [2], 1: [2], 2: [2]},
                        3: {0: [3], 1: [3], 2: [3]},
                        4: {0: [4], 1: [4], 2: [4]},
                        5: {0: [5], 1: [5], 2: [5]},
                        6: {0: [6], 1: [6], 2: [6]},
                        7: {0: [7], 1: [7], 2: [7]},

        }

        rc = reachability_clousure(net, max_hops_route_in_ts=0)
        self.assertEqual(len(expected_rc.keys()), len(rc.keys()))
        self.assertTrue(all(k in rc.keys() for k in expected_rc.keys()))
        for k in expected_rc.keys():
            self.assertEqual(len(expected_rc[k].keys()), len(rc[k].keys()))
            for ts in range(net.num_of_ts):
                self.assertEqual(len(expected_rc[k][ts]), len(rc[k][ts]))
                self.assertTrue(all(n in rc[k][ts] for n in expected_rc[k][ts]))

if __name__ == '__main__':
    unittest.main()
