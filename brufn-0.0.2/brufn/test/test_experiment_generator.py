import unittest
from brufn.experiment_generator import generate_omnet_traffic
class Test_Experiment_Generator(unittest.TestCase):

    def test_generate_omnet_traffic1(self):
        traffic = generate_omnet_traffic({1:[2,3]})
        expected_traffic=(
                            'dtnsim.node[1].app.enable=true\n'
                            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
                            'dtnsim.node[1].app.start="0,0"\n'
                            'dtnsim.node[1].app.destinationEid="2,3"\n'
                            'dtnsim.node[1].app.size="1,1"\n\n'
                        )
        self.assertEqual(expected_traffic, traffic)

    def test_generate_omnet_traffic2(self):
        traffic = generate_omnet_traffic({1:[2,3], 10:[4,5,6]})
        expected_traffic=(
                            'dtnsim.node[1].app.enable=true\n'
                            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
                            'dtnsim.node[1].app.start="0,0"\n'
                            'dtnsim.node[1].app.destinationEid="2,3"\n'
                            'dtnsim.node[1].app.size="1,1"\n\n'
                            
                            'dtnsim.node[10].app.enable=true\n'
                            'dtnsim.node[10].app.bundlesNumber="1,1,1"\n'
                            'dtnsim.node[10].app.start="0,0,0"\n'
                            'dtnsim.node[10].app.destinationEid="4,5,6"\n'
                            'dtnsim.node[10].app.size="1,1,1"\n\n'
                        )
        self.assertEqual(expected_traffic, traffic)

    def test_generate_omnet_traffic3(self):
        traffic = generate_omnet_traffic({1:[2,3]}, traffic_startt={1:{3:10}})
        expected_traffic=(
                            'dtnsim.node[1].app.enable=true\n'
                            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
                            'dtnsim.node[1].app.start="0,10"\n'
                            'dtnsim.node[1].app.destinationEid="2,3"\n'
                            'dtnsim.node[1].app.size="1,1"\n\n'
                        )
        self.assertEqual(expected_traffic, traffic)

    def test_generate_omnet_traffic4(self):
        traffic = generate_omnet_traffic({1: [2, 3], 10: [4, 5, 6]}, traffic_startt={1:{2:10}, 10:{4:1,5:2,6:30}})
        expected_traffic = (
            'dtnsim.node[1].app.enable=true\n'
            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
            'dtnsim.node[1].app.start="10,0"\n'
            'dtnsim.node[1].app.destinationEid="2,3"\n'
            'dtnsim.node[1].app.size="1,1"\n\n'

            'dtnsim.node[10].app.enable=true\n'
            'dtnsim.node[10].app.bundlesNumber="1,1,1"\n'
            'dtnsim.node[10].app.start="1,2,30"\n'
            'dtnsim.node[10].app.destinationEid="4,5,6"\n'
            'dtnsim.node[10].app.size="1,1,1"\n\n'
        )
        self.assertEqual(expected_traffic, traffic)

    def test_generate_omnet_traffic5(self):
        traffic = generate_omnet_traffic({1: [2, 3], 10: [4, 5, 6]},
                                         traffic_ttls={1: 100, 10: 30})
        expected_traffic = (
            'dtnsim.node[1].app.enable=true\n'
            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
            'dtnsim.node[1].app.start="0,0"\n'
            'dtnsim.node[1].app.destinationEid="2,3"\n'
            'dtnsim.node[1].app.size="1,1"\n'
            'dtnsim.node[1].app.ttl=100\n\n'

            'dtnsim.node[10].app.enable=true\n'
            'dtnsim.node[10].app.bundlesNumber="1,1,1"\n'
            'dtnsim.node[10].app.start="0,0,0"\n'
            'dtnsim.node[10].app.destinationEid="4,5,6"\n'
            'dtnsim.node[10].app.size="1,1,1"\n'
            'dtnsim.node[10].app.ttl=30\n\n'
        )
        self.assertEqual(expected_traffic, traffic)


    def test_generate_omnet_traffic6(self):
        traffic = generate_omnet_traffic({1: [2, 3], 10: [4, 5, 6]}, traffic_startt={1:{2:10}, 10:{4:1,5:2,6:30}},
                                         traffic_ttls={1: 10, 10: 14})
        expected_traffic = (
            'dtnsim.node[1].app.enable=true\n'
            'dtnsim.node[1].app.bundlesNumber="1,1"\n'
            'dtnsim.node[1].app.start="10,0"\n'
            'dtnsim.node[1].app.destinationEid="2,3"\n'
            'dtnsim.node[1].app.size="1,1"\n'
            'dtnsim.node[1].app.ttl=10\n\n'

            'dtnsim.node[10].app.enable=true\n'
            'dtnsim.node[10].app.bundlesNumber="1,1,1"\n'
            'dtnsim.node[10].app.start="1,2,30"\n'
            'dtnsim.node[10].app.destinationEid="4,5,6"\n'
            'dtnsim.node[10].app.size="1,1,1"\n'
            'dtnsim.node[10].app.ttl=14\n\n'
        )
        self.assertEqual(expected_traffic, traffic)

if __name__ == '__main__':
    unittest.main()