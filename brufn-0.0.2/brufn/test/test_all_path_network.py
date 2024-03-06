import unittest
from brufn.network import Net, Contact
from typing import List
import os


class TestAllPath(unittest.TestCase):
    '''
        Auxiliary method for testing TestGetAllTimeVaryingPaths and TestGetAllTimeVaryingPathsNonRevisiting
    '''

    def setUp(self):
        cwd = os.getcwd()
        print(cwd)
        if cwd.endswith('brufn'):
            self.path = 'test'
        else:
            self.path = ''

    def are_equal_paths(self, p1:List[Contact], p2:List[Contact])->bool:
        if len(p1) == len(p2):
            for c_p1, c_p2 in zip(p1,p2):
                if c_p1.from_ != c_p2.from_ or c_p1.to != c_p2.to or c_p1.ts != c_p1.ts \
                        or c_p1.begin_time != c_p2.begin_time or c_p1.end_time != c_p2.end_time:
                    return False
            return True
        else:
            return False

    def gen_binomial_net(self, levels):
        assert levels > 0, "Levels must be greater than 1"
        contacts = []
        for level in range(1, levels-1):
            target = int(2**level - 1)
            for node in range(int(2**(level - 1) - 1), int(2**level - 1)):
                contacts.append(Contact(node, target, 0))
                target += 1
                contacts.append(Contact(node, target, 0))
                target += 1

        # Link nodes levels - 2 to target node
        for node in range(int(2 ** (levels - 2) - 1), int(2 ** (levels - 1) - 1)):
            contacts.append(Contact(node, 2 ** (levels - 1) - 1, 0))

        return Net(2 ** (levels - 1), contacts)

    def gen_binomial_net_level_by_ts(self, levels):
        assert levels > 0, "Levels must be greater than 1"
        contacts = []
        for level in range(1, levels - 1):
            target = int(2 ** level - 1)
            for node in range(int(2 ** (level - 1) - 1), int(2 ** level - 1)):
                contacts.append(Contact(node, target, level-1))
                target += 1
                contacts.append(Contact(node, target, level-1))
                target += 1

        # Link nodes levels - 2 to target node
        for node in range(int(2**(levels-2) - 1), int(2**(levels-1) - 1)):
            contacts.append(Contact(node, 2**(levels-1) - 1, levels-2))

        return Net(2**(levels - 1), contacts)

class TestGetAllTimeVaryingPaths(TestAllPath):

    def test_gen_all_paths_1(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/twopath_twots.py'))
        #net.to_dot('test/net_test_all_paths/', 'twopath_twots.dot')
        verified_path = [[Contact(0,1,0), Contact(1,3,1)],
                         [Contact(0, 2, 0), Contact(2, 3, 1)]
                         ]
        paths = net.get_all_time_varying_paths(0,3,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")

    def test_gen_all_paths_2(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/net1.py'), contact_pf_required=False)
        verified_path = [[Contact(0,1,0), Contact(1,3,0), Contact(3,4,0)],
                         [Contact(0, 2, 0), Contact(2, 3, 0), Contact(3, 4, 0)],
                         [Contact(0, 1, 0), Contact(1, 2, 0), Contact(2, 3, 0),  Contact(3, 4, 0)],
                         [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 3, 0), Contact(3, 4, 0)],
                         ]
        paths = net.get_all_time_varying_paths(0,4,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")


    def test_gen_all_paths_3(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/net1.py'), contact_pf_required=False)
        verified_path = [[Contact(0,1,0), Contact(1,3,0), Contact(3,4,0)],
                         [Contact(0, 2, 0), Contact(2, 3, 0), Contact(3, 4, 0)],
                         [Contact(0, 1, 0), Contact(1, 2, 0), Contact(2, 3, 0),  Contact(3, 4, 0)],
                         [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 3, 0), Contact(3, 4, 0)],
                         ]
        paths = net.get_all_time_varying_paths(0,4,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")

    def test_gen_all_paths_bin_4levels(self):
        levels = 4; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_gen_all_paths_bin_5levels(self):
        levels = 5; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_6levels(self):
        levels = 6; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_2levels_2ts(self):
        levels = 2; num_ts = 2
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_3levels_3ts(self):
        levels = 3; num_ts = 3
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_4levels_4ts(self):
        levels = 4; num_ts = 4
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_gen_all_paths_bin_5levels_5ts(self):
        levels = 5; num_ts = 4
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_6levels_6ts(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_gen_all_paths_bin_6levels_6ts_varying_starting_ts_1(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, level)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_6levels_6ts_varying_starting_ts_2(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [0,0,0,0,0,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, level+1)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_net0_ts8_reduced(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net0_ts8_reduced.py'), contact_pf_required=False)
        verified_paths = [ [Contact(0,7,1)],
                           [Contact(0,1,0), Contact(1,0,1), Contact(0,7,1)],
                           [Contact(0, 2, 0), Contact(2,1,0), Contact(1, 0, 1), Contact(0, 7, 1)]
        ]
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net0_ts8_reduced.dot')
        paths = net.get_all_time_varying_paths(0, 7, 0)
        self.assertEqual(len(verified_paths), len(paths), f"There must be {len(verified_paths)} but there are {len(paths)}")
        for p1 in verified_paths:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")

    def test_netstorage(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net_storage.py'), contact_pf_required=False)
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net_storage.dot')
        paths = net.get_all_time_varying_paths(0,2,0)
        self.assertEqual(1, len(paths))
        self.assertTrue(self.are_equal_paths([Contact(0,1,0),Contact(1,2,5)],paths[0]))

    def test_net0_ts8(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net0_ts8.py'), contact_pf_required=False)
        verified_paths = [ [Contact(0,7,1)],
                           [Contact(0,1,0), Contact(1,0,1), Contact(0,7,1)],
                           [Contact(0, 2, 0), Contact(2,1,0), Contact(1, 0, 1), Contact(0, 7, 1)],
                           [Contact(0, 1, 0), Contact(1, 6, 1), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 0), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 1), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 6, 1), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 0), Contact(6, 1, 1), Contact(1, 0, 1), Contact(0, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 1), Contact(6, 1, 1), Contact(1, 0, 1), Contact(0, 7, 1)],
        ]
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net0_ts8.dot')
        paths = net.get_all_time_varying_paths(0, 7, 0)
        self.assertEqual(len(verified_paths), len(paths), f"There must be {len(verified_paths)} but there are {len(paths)}")
        for p1 in verified_paths:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")


class TestGetAllTimeVaryingPathsNonRevisiting(TestAllPath):

    def test_gen_all_paths_1(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/twopath_twots.py'))
        #net.to_dot('test/net_test_all_paths/', 'twopath_twots.dot')
        verified_path = [[Contact(0,1,0), Contact(1,3,1)],
                         [Contact(0, 2, 0), Contact(2, 3, 1)]
                         ]
        paths = net.get_all_time_varying_paths_non_revising(0,3,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")

    def test_gen_all_paths_2(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/net1.py'), contact_pf_required=False)
        verified_path = [[Contact(0,1,0), Contact(1,3,0), Contact(3,4,0)],
                         [Contact(0, 2, 0), Contact(2, 3, 0), Contact(3, 4, 0)],
                         [Contact(0, 1, 0), Contact(1, 2, 0), Contact(2, 3, 0),  Contact(3, 4, 0)],
                         [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 3, 0), Contact(3, 4, 0)],
                         ]
        paths = net.get_all_time_varying_paths_non_revising(0,4,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")


    def test_gen_all_paths_3(self):
        net = Net.get_net_from_file(os.path.join(self.path,'net_test_all_paths/net1.py'), contact_pf_required=False)
        verified_path = [[Contact(0,1,0), Contact(1,3,0), Contact(3,4,0)],
                         [Contact(0, 2, 0), Contact(2, 3, 0), Contact(3, 4, 0)],
                         [Contact(0, 1, 0), Contact(1, 2, 0), Contact(2, 3, 0),  Contact(3, 4, 0)],
                         [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 3, 0), Contact(3, 4, 0)],
                         ]
        paths = net.get_all_time_varying_paths_non_revising(0,4,0)
        self.assertEqual(len(verified_path), len(paths), f"There must be {len(verified_path)} but there are {len(paths)}")
        for p1 in verified_path:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")


    def test_gen_all_paths_bin_4levels(self):
        levels = 4; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_5levels(self):
        levels = 5; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")
    def test_gen_all_paths_bin_6levels(self):
        levels = 6; num_ts = 1
        net = self.gen_binomial_net(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_2levels_2ts(self):
        levels = 2; num_ts = 2
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_3levels_3ts(self):
        levels = 3; num_ts = 3
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_4levels_4ts(self):
        levels = 4; num_ts = 4
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                #print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        #print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_gen_all_paths_bin_5levels_5ts(self):
        levels = 5; num_ts = 4
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_6levels_6ts(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_gen_all_paths_bin_6levels_6ts_varying_starting_ts_1(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [16,8,4,2,1,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, level)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")

    def test_gen_all_paths_bin_6levels_6ts_varying_starting_ts_2(self):
        levels = 6; num_ts = 6
        net = self.gen_binomial_net_level_by_ts(levels)
        path_number_by_level = [0,0,0,0,0,0]

        #net.print_to_file('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.py')
        #net.to_dot('test/net_test_all_paths', file_name=f'binomial_levels={levels}-num_ts={num_ts}.dot')

        source = 0
        for level in range(levels - 1):
            for n in range(2**level):
                paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, level+1)
                self.assertEqual(path_number_by_level[level], len(paths),
                                 f"Nodes from level {level} must have {path_number_by_level[level]} each but they have {len(paths)}")
                print("Source=", source)
                source += 1

        source = 2**(levels-1) - 1
        print("Source=", source)
        paths = net.get_all_time_varying_paths_non_revising(source, 2**(levels-1) - 1, 0)
        self.assertEqual(path_number_by_level[levels-1], len(paths),
                         f"Nodes from level {levels-1} must have {path_number_by_level[levels-1]} each but they have {len(paths)}")


    def test_net0_ts8_reduced(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net0_ts8_reduced.py'), contact_pf_required=False)
        verified_paths = [[Contact(0,7,1)]
                           #[Contact(0,1,0), Contact(1,0,1), Contact(0,7,1)], # Revisiting node 0
                           #[Contact(0, 2, 0), Contact(2,1,0), Contact(1, 0, 1), Contact(0, 7, 1)] # Revisiting node 0
        ]
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net0_ts8_reduced.dot')
        paths = net.get_all_time_varying_paths_non_revising(0, 7, 0)
        self.assertEqual(len(verified_paths), len(paths), f"There must be {len(verified_paths)} but there are {len(paths)}")
        for p1 in verified_paths:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")
    def test_netstorage(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net_storage.py'), contact_pf_required=False)
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net_storage.dot')
        paths = net.get_all_time_varying_paths_non_revising(0,2,0)
        self.assertEqual(1, len(paths))
        self.assertTrue(self.are_equal_paths([Contact(0,1,0),Contact(1,2,5)],paths[0]))

    def test_net0_ts8(self):
        net = Net.get_net_from_file(os.path.join(self.path, 'net_test_all_paths/net0_ts8.py'), contact_pf_required=False)
        verified_paths = [ [Contact(0,7,1)],
                           #[Contact(0,1,0), Contact(1,0,1), Contact(0,7,1)], Revisiting node 0
                           #[Contact(0, 2, 0), Contact(2,1,0), Contact(1, 0, 1), Contact(0, 7, 1)], Revisiting node 0
                           [Contact(0, 1, 0), Contact(1, 6, 1), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 0), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 6, 1), Contact(6, 7, 1)],
                           [Contact(0, 2, 0), Contact(2, 1, 0), Contact(1, 6, 1), Contact(6, 7, 1)],
                           #[Contact(0, 2, 0), Contact(2, 6, 0), Contact(6, 1, 1), Contact(1, 0, 1), Contact(0, 7, 1)], Revisiting node 0
                           #[Contact(0, 2, 0), Contact(2, 6, 1), Contact(6, 1, 1), Contact(1, 0, 1), Contact(0, 7, 1)], Revisiting node 0
        ]
        net.to_dot(os.path.join(self.path, 'net_test_all_paths'),file_name='net0_ts8.dot')
        paths = net.get_all_time_varying_paths_non_revising(0, 7, 0)
        self.assertEqual(len(verified_paths), len(paths), f"There must be {len(verified_paths)} but there are {len(paths)}")
        for p1 in verified_paths:
            found = False
            for p2 in paths:
                if self.are_equal_paths(p1, p2):
                    found = True
                    break
            self.assertTrue(found, f"Path {'->'.join([str(c) for c in p1])} was not found")



if __name__ == '__main__':
    unittest.main()