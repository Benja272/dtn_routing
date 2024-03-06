import unittest
from brufn.contact_plan import CP_Contact, ContactPlan, GraphTS
from brufn.network import Contact
import os
import shutil

class Test_Contact(unittest.TestCase):

    def test_get_contact_from_str_1(self):
        contact = 'a contact +0000011 +0000254 34 111 1'

        parse_contact = CP_Contact.get_contact_from_str(contact, 10)
        self.assertEqual(parse_contact.source, 34)
        self.assertEqual(parse_contact.target, 111)
        self.assertEqual(parse_contact.start_t, 11)
        self.assertEqual(parse_contact.end_t, 254)
        self.assertEqual(parse_contact.data_rate, 1)
        self.assertEqual(parse_contact.id, 10)

    def test_get_contact_from_str_2(self):
        contact = 'a contact +0000011 +0000254 34 111 1'

        parse_contact = CP_Contact.get_contact_from_str(contact)
        self.assertEqual(parse_contact.source, 34)
        self.assertEqual(parse_contact.target, 111)
        self.assertEqual(parse_contact.start_t, 11)
        self.assertEqual(parse_contact.end_t, 254)
        self.assertEqual(parse_contact.data_rate, 1)
        self.assertEqual(parse_contact.id, -1)

    def test_contact_to_str_1(self):
        contact = 'a contact +0000011 +0000254 34 111 1'
        self.assertEqual(str(CP_Contact.get_contact_from_str(contact)), contact)


    def test_contact_to_str_2(self):
        contact = 'a contact +8000011 +9000254 34 111 1'
        self.assertEqual(str(CP_Contact.get_contact_from_str(contact)), contact)


class Test_ContactPlan(unittest.TestCase):
    def setUp(self):
        os.makedirs('output', exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree('output')

    def test_parse_contact_plan1(self):
        cp_input_path = 'cp/cp1.txt'
        cp_output_path = 'output/cp_output.txt'
        cp = ContactPlan.parse_contact_plan(cp_input_path)
        cp.print_to_file(cp_output_path)
        with open(cp_input_path, 'r') as cp_input, open(cp_output_path, 'r') as cp_output:
            for l1, l2 in zip(cp_output.readlines(), cp_input.readlines()):
                self.assertEqual(l1, l2)

    def test_parse_contact_plan2(self):
        cp_input_path = 'cp/RingRoad_16sats_Walker_6hotspots_simtime24hs_comrange1000km.txt'
        cp_output_path = 'output/cp_output.txt'
        cp = ContactPlan.parse_contact_plan(cp_input_path)
        cp.print_to_file(cp_output_path)
        with open(cp_output_path, 'r') as cp_output:
            id = 1;
            cp_lines = cp_output.readlines()
            self.assertEqual(len(cp.contacts), len(cp_lines))
            for l1, c2 in zip(cp_lines, cp.contacts):
                c1 = CP_Contact.get_contact_from_str(l1, id)
                self.assertEqual(c1.source, c2.source)
                self.assertEqual(c1.target, c2.target)
                self.assertEqual(c1.start_t, c2.start_t)
                self.assertEqual(c1.end_t, c2.end_t)
                self.assertEqual(c1.data_rate, c2.data_rate)
                self.assertEqual(c1.data_rate, c2.data_rate)
                self.assertEqual(c1.id, c2.id)
                id += 1

    def test_rename_eids(self):
        contacts = ['a contact +0000011 +0000254 10 111 1', 'a contact +0000011 +0000254 20 30 1', 'a contact +0000011 +0000254 11 1 1', 'a contact +0000011 +0000254 1 10 1']
        expected_renamed_contacts = ['a contact +0000011 +0000254 100 111 1', 'a contact +0000011 +0000254 20 30 1', 'a contact +0000011 +0000254 11 10 1', 'a contact +0000011 +0000254 10 100 1']
        contacts = [CP_Contact.get_contact_from_str(c) for c in contacts]
        cp = ContactPlan(contacts)
        cp.rename_eids({10:100, 1:10})
        self.assertEqual([str(c) for c in cp.contacts], expected_renamed_contacts)


    def test_slice_cp_1(self):
        contacts = [
                    'a contact +0000010 +0000020 9 112 1',
                    'a contact +0000020 +0000030 31 114 1',
                    'a contact +0000040 +0000050 50 119 1',
                    'a contact +0000050 +0000060 112 9 1',
                    'a contact +0000060 +0000070 114 31 1',
                    'a contact +0000070 +0000080 119 50 1'
        ]
        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp_slice = cp.slice_cp(40, cp.end_time)
        with open('cp/cp1.txt', 'r') as cp_input:
            for i, c_str in enumerate(contacts[2:]):
                self.assertEqual(str(cp_slice.contacts[i]), c_str)

    def test_slice_cp_2(self):
        contacts = [
                    'a contact +0000010 +0000020 9 112 1',
                    'a contact +0000020 +0000030 31 114 1',
                    'a contact +0000040 +0000050 50 119 1',
                    'a contact +0000050 +0000060 112 9 1',
                    'a contact +0000060 +0000070 114 31 1',
                    'a contact +0000070 +0000080 119 50 1'
        ]
        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp_slice = cp.slice_cp(40, 70)
        for i, c_str in enumerate(contacts[2:-1]):
            self.assertEqual(str(cp_slice.contacts[i]), c_str)

    def test_slice_cp_3(self):
        contacts = [
                    'a contact +0000010 +0000020 9 112 1',
                    'a contact +0000020 +0000030 31 114 1',
                    'a contact +0000040 +0000050 50 119 1',
                    'a contact +0000050 +0000060 112 9 1',
                    'a contact +0000060 +0000070 114 31 1',
                    'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = [
                    'a contact +0000025 +0000030 31 114 1',
                    'a contact +0000040 +0000050 50 119 1',
                    'a contact +0000050 +0000060 112 9 1',
                    'a contact +0000060 +0000065 114 31 1',
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp_slice = cp.slice_cp(25, 65)
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp_slice.contacts[i]), c_str)

        def test_slice_cp_4(self):
            contacts = [
                'a contact +0000000 +0000010 9 112 1',
                'a contact +0000010 +0000020 9 112 1',
                'a contact +0000020 +0000030 31 114 1',
                'a contact +0000040 +0000050 50 119 1',
                'a contact +0000050 +0000060 112 9 1',
                'a contact +0000060 +0000070 114 31 1',
                'a contact +0000070 +0000080 119 50 1'
            ]

            cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
            cp_slice = cp.slice_cp(cp.start_time, 60)
            for i, c_str in enumerate(contacts[:-2]):
                self.assertEqual(str(cp_slice.contacts[i]), c_str)

    def test_slice_cp_5(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1',
            'a contact +0000010 +0000020 9 13 1',
        ]
        expected_contacts = [
            'a contact +0000001 +0000010 9 112 1',
            'a contact +0000010 +0000019 9 13 1',
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c, id=i+1) for i,c in enumerate(contacts)])
        cp_slice = cp.slice_cp(1, 19)
        for i, c in enumerate(cp.contacts):
            self.assertEqual(i+1, c.id)

        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(c_str, str(cp_slice.contacts[i]))
            self.assertEqual(i+1, cp_slice.contacts[i].id)


    def test_slice_cp_6(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = []

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.slice_cp(80, 150)
        self.assertEqual(0, cp.node_number)
        self.assertEqual(-1, cp.start_time)
        self.assertEqual(-1, cp.end_time)

    def test_filter_contacts_1(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = contacts

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp.filter_contact_by_endpoints([9, 31, 50, 112, 114, 119], [112, 13, 114, 119, 9, 31, 50])
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp.contacts[i]), c_str)

    def test_filter_contacts_2(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = []

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.filter_contact_by_endpoints(range(8), range(8))
        self.assertEqual(cp.contacts, expected_contacts)

    def test_filter_contacts_3(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = [
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.filter_contact_by_endpoints([31, 50, 112, 114, 119], [112, 13, 114, 119, 9, 31, 50])
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp.contacts[i]), c_str)

    def test_filter_contacts_4(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.filter_contact_by_endpoints([9, 31, 50, 112, 114, 119], [112, 13, 114, 119, 31, 50])
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp.contacts[i]), c_str)

    def test_filter_contacts_5(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000050 +0000060 112 9 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]
        expected_contacts = [
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1'
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.filter_contact_by_endpoints([31, 50, 112, 114, 119], [112, 13, 114, 119, 31, 50])
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp.contacts[i]), c_str)


    def test_filter_contacts_6(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000040 +0000050 50 119 1',
            'a contact +0000060 +0000070 114 31 1',
            'a contact +0000070 +0000080 119 50 1',
            'a contact +0000080 +0000090 112 15 1',
        ]
        expected_contacts =[
            'a contact +0000020 +0000030 31 114 1',
            'a contact +0000070 +0000080 119 50 1',
            'a contact +0000080 +0000090 112 15 1',
        ]
        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        cp = cp.filter_contact_by_endpoints([31, 50, 112, 119], [112, 13, 114, 50, 15])
        for i, c_str in enumerate(expected_contacts):
            self.assertEqual(str(cp.contacts[i]), c_str)


    def test__get_net(self):
        contacts = [
            'a contact +0000000 +0000010 9 112 1',
            'a contact +0000010 +0000020 9 13 1',
            'a contact +0000022 +0000030 31 114 1',
        ]
        expected_contacts_net = [
            Contact(8, 111, 0, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 1, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 2, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 3, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 4, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 5, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 6, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 7, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 8, identifier=0, begin_time=0, end_time=10),
            Contact(8, 111, 9, identifier=0, begin_time=0, end_time=10),

            Contact(8, 12, 10, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 11, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 12, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 13, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 14, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 15, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 16, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 17, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 18, identifier=1, begin_time=10, end_time=20),
            Contact(8, 12, 19, identifier=1, begin_time=10, end_time=20),

            Contact(30, 113, 22, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 23, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 24, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 25, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 26, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 27, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 28, identifier=2, begin_time=22, end_time=30),
            Contact(30, 113, 29, identifier=2, begin_time=22, end_time=30),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        self.assertTrue(114, cp.node_number)
        net = cp._get_net(lambda l: l, delta_t = 1, output_path='output')

        self.assertEqual(114, net.num_of_nodes)
        self.assertEqual(30, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts), f"{c} not generated")

    def test_contact_eq(self):
        self.assertEqual(CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'), CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'))
        self.assertNotEqual(CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'), CP_Contact.get_contact_from_str('a contact +0000030 +0000040 31 114 1'))
        self.assertNotEqual(CP_Contact.get_contact_from_str('a contact +0000020 +0000030 32 114 1'), CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'))
        self.assertNotEqual(CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 115 1'), CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'))
        self.assertNotEqual(CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 2'), CP_Contact.get_contact_from_str('a contact +0000020 +0000030 31 114 1'))

    def test_generate_slicing_abstraction1(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=1),
            Contact(0, 1, 1, identifier=1, begin_time=1, end_time=2),
            Contact(0, 1, 2, identifier=2, begin_time=2, end_time=3),
            Contact(0, 1, 3, identifier=3, begin_time=3, end_time=4),
            Contact(0, 1, 4, identifier=5, begin_time=4, end_time=5),
            Contact(0, 1, 5, identifier=7, begin_time=5, end_time=6),
            Contact(0, 1, 6, identifier=9, begin_time=6, end_time=7),
            Contact(0, 1, 7, identifier=11, begin_time=7, end_time=8),
            Contact(0, 1, 8, identifier=13, begin_time=8, end_time=9),
            Contact(0, 1, 9, identifier=15, begin_time=9, end_time=10),

            Contact(1, 2, 3, identifier=4, begin_time=3, end_time=4),
            Contact(1, 2, 4, identifier=6, begin_time=4, end_time=5),
            Contact(1, 2, 5, identifier=8, begin_time=5, end_time=6),
            Contact(1, 2, 6, identifier=10, begin_time=6, end_time=7),
            Contact(1, 2, 7, identifier=12, begin_time=7, end_time=8),
            Contact(1, 2, 8, identifier=14, begin_time=8, end_time=9),
            Contact(1, 2, 9, identifier=16, begin_time=9, end_time=10),
            Contact(1, 2, 10, identifier=17, begin_time=10, end_time=11),
            Contact(1, 2, 11, identifier=18, begin_time=11, end_time=12),
            Contact(1, 2, 12, identifier=19, begin_time=12, end_time=13),

            Contact(2, 3, 13, identifier=20, begin_time=13, end_time=14),
            Contact(2, 3, 14, identifier=21, begin_time=14, end_time=15),
            Contact(2, 3, 15, identifier=22, begin_time=15, end_time=16),
            Contact(2, 3, 16, identifier=23, begin_time=16, end_time=17),
            Contact(2, 3, 17, identifier=24, begin_time=17, end_time=18),
            Contact(2, 3, 18, identifier=25, begin_time=18, end_time=19),
            Contact(2, 3, 19, identifier=26, begin_time=19, end_time=20),
            Contact(2, 3, 20, identifier=27, begin_time=20, end_time=21),
            Contact(2, 3, 21, identifier=28, begin_time=21, end_time=22),
            Contact(2, 3, 22, identifier=29, begin_time=22, end_time=23),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_slicing_abstraction(slicing_time=1, min_contact_duration=1)

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(23, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.id == cc.id and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")


    def test_generate_slicing_abstraction2(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=2),
            Contact(0, 1, 1, identifier=1, begin_time=2, end_time=4),
            Contact(0, 1, 2, identifier=3, begin_time=4, end_time=6),
            Contact(0, 1, 3, identifier=5, begin_time=6, end_time=8),
            Contact(0, 1, 4, identifier=7, begin_time=8, end_time=10),

            Contact(1, 2, 1, identifier=2, begin_time=3, end_time=4),
            Contact(1, 2, 2, identifier=4, begin_time=4, end_time=6),
            Contact(1, 2, 3, identifier=6, begin_time=6, end_time=8),
            Contact(1, 2, 4, identifier=8, begin_time=8, end_time=10),
            Contact(1, 2, 5, identifier=9, begin_time=10, end_time=12),
            Contact(1, 2, 6, identifier=10, begin_time=12, end_time=13),

            Contact(2, 3, 6, identifier=11, begin_time=13, end_time=14),
            Contact(2, 3, 7, identifier=12, begin_time=14, end_time=16),
            Contact(2, 3, 8, identifier=13, begin_time=16, end_time=18),
            Contact(2, 3, 9, identifier=14, begin_time=18, end_time=20),
            Contact(2, 3, 10, identifier=15, begin_time=20, end_time=22),
            Contact(2, 3, 11, identifier=16, begin_time=22, end_time=23),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c, id=id+1) for id, c in enumerate(contacts)])
        net = cp.generate_slicing_abstraction(slicing_time=2, min_contact_duration=1)

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(12, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.id == cc.id and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")

    def test_generate_slicing_abstraction3(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=2),
            Contact(0, 1, 1, identifier=1, begin_time=2, end_time=4),
            Contact(0, 1, 2, identifier=2, begin_time=4, end_time=6),
            Contact(0, 1, 3, identifier=4, begin_time=6, end_time=8),
            Contact(0, 1, 4, identifier=6, begin_time=8, end_time=10),

            Contact(1, 2, 2, identifier=3, begin_time=4, end_time=6),
            Contact(1, 2, 3, identifier=5, begin_time=6, end_time=8),
            Contact(1, 2, 4, identifier=7, begin_time=8, end_time=10),
            Contact(1, 2, 5, identifier=8, begin_time=10, end_time=12),

            Contact(2, 3, 7, identifier=9, begin_time=14, end_time=16),
            Contact(2, 3, 8, identifier=10, begin_time=16, end_time=18),
            Contact(2, 3, 9, identifier=11, begin_time=18, end_time=20),
            Contact(2, 3, 10, identifier=12, begin_time=20, end_time=22),
        ]


        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_slicing_abstraction(slicing_time=2, min_contact_duration=2)

        self.assertEqual(4, net.num_of_nodes, )
        self.assertEqual(11, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.id == cc.id and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")

    def test_generate_slicing_abstraction4(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=5),
            Contact(0, 1, 1, identifier=2, begin_time=5, end_time=10),

            Contact(1, 2, 0, identifier=1, begin_time=3, end_time=5),
            Contact(1, 2, 1, identifier=3, begin_time=5, end_time=10),
            Contact(1, 2, 2, identifier=4, begin_time=10, end_time=13),

            Contact(2, 3, 2, identifier=5, begin_time=13, end_time=15),
            Contact(2, 3, 3, identifier=6, begin_time=15, end_time=20),
            Contact(2, 3, 4, identifier=7, begin_time=20, end_time=23),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_slicing_abstraction(slicing_time=5, min_contact_duration=2)

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(5, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.id == cc.id and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")



    def test_generate_slicing_abstraction5(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=5),
            Contact(0, 1, 1, identifier=1, begin_time=5, end_time=10),

            Contact(1, 2, 1, identifier=2, begin_time=5, end_time=10),
            Contact(1, 2, 2, identifier=3, begin_time=10, end_time=13),

            Contact(2, 3, 3, identifier=4, begin_time=15, end_time=20),
            Contact(2, 3, 4, identifier=5, begin_time=20, end_time=23),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_slicing_abstraction(slicing_time=5, min_contact_duration=3)

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(5, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.id == cc.id and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")

    def test_generate_all_contact_abstraction1(self):
        contacts = [
            'a contact +0000000 +0000010 1 2 1',
            'a contact +0000003 +0000013 2 3 1',
            'a contact +0000013 +0000023 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 1, 0, identifier=0, begin_time=0, end_time=10),
            Contact(1, 2, 0, identifier=0, begin_time=3, end_time=13),

            Contact(2, 3, 1, identifier=0, begin_time=13, end_time=23),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_all_contact_abstraction()

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(2, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.begin_time == cc.begin_time and c.end_time == cc.end_time), f"{c} not generated")

    def test_generate_all_contact_abstraction2(self):
        contacts = [
            'a contact +0000010 +0000020 1 4 1',
            'a contact +0000010 +0000020 4 1 1',

            'a contact +0000020 +0000030 2 4 1',
            'a contact +0000020 +0000030 4 2 1',

            'a contact +0000020 +0000040 2 3 1',
            'a contact +0000020 +0000040 3 2 1',

            'a contact +0000030 +0000060 3 4 1',
            'a contact +0000030 +0000060 4 3 1',
        ]
        expected_contacts_net = [
            Contact(0, 3, 0, identifier=0, begin_time=10, end_time=20),
            Contact(3, 0, 0, identifier=0, begin_time=10, end_time=20),

            Contact(1, 3, 1, identifier=0, begin_time=20, end_time=30),
            Contact(3, 1, 1, identifier=0, begin_time=20, end_time=30),

            Contact(1, 2, 1, identifier=0, begin_time=20, end_time=40),
            Contact(2, 1, 1, identifier=0, begin_time=20, end_time=40),

            Contact(2, 3, 1, identifier=0, begin_time=30, end_time=60),
            Contact(3, 2, 1, identifier=0, begin_time=30, end_time=60),

        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_all_contact_abstraction()

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(2, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")

    def test_generate_all_contact_abstraction3(self):
        contacts = [
            'a contact +0000010 +0000020 1 4 1',

            'a contact +0000030 +0000040 2 4 1',

            'a contact +0000020 +0000040 2 3 1',

            'a contact +0000030 +0000060 3 4 1',
        ]
        expected_contacts_net = [
            Contact(0, 3, 0, identifier=0, begin_time=10, end_time=20),

            Contact(1, 3, 1, identifier=0, begin_time=30, end_time=40),

            Contact(1, 2, 1, identifier=0, begin_time=20, end_time=40),

            Contact(2, 3, 1, identifier=0, begin_time=30, end_time=60),
        ]

        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        net = cp.generate_all_contact_abstraction()

        self.assertEqual(4, net.num_of_nodes)
        self.assertEqual(2, net.num_of_ts)
        self.assertEqual(len(expected_contacts_net), len(net.contacts))
        for c in expected_contacts_net:
            self.assertTrue(any(cc for cc in net.contacts if c.from_ == cc.from_ and c.to == cc.to and c.ts == cc.ts and c.begin_time==cc.begin_time and c.end_time==cc.end_time), f"{c} not generated")

    def test_CP_Contact__eq__method__raise_exeption(self):
        c = CP_Contact.get_contact_from_str('a contact +0000010 +0000020 1 3 1')
        self.assertRaises(ValueError, c.__eq__, [])

    def test_GraphTS__eq__method__raise_exeption(self):
        graph = GraphTS(0,20)
        self.assertRaises(ValueError, graph.__eq__, [])

    def test_generate_statistics(self):
        contacts = [
            'a contact +0000010 +0000020 1 4 1',

            'a contact +0000030 +0000040 2 4 1',

            'a contact +0000020 +0000040 2 3 1',

            'a contact +0000030 +0000060 3 4 1',
        ]
        cp = ContactPlan([CP_Contact.get_contact_from_str(c) for c in contacts])
        statistics = cp.generate_statistics()
        self.assertEqual(statistics['contact_duration-avg'], 17.5)
        self.assertEqual(statistics['contact_duration-min'], 10)
        self.assertEqual(statistics['contact_duration-max'], 30)

if __name__ == '__main__':
    unittest.main()
