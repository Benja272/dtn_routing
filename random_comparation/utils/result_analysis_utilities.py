def get_success_pr_from_modest_file(fpath:str)->float:
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                if 'mcsta' in fpath:
                    if l.startswith('  Probability: '):
                        return float(l[len('  Probability: '):])
                else:
                    if l.startswith('  Estimated max. probability: '):
                        return float(l[len('  Estimated max. probability: '):])
            
            assert False, f"fun get_success_pr_from_modest_file fails to extract success pr from {fpath} "


def get_success_pr_from_modest_uniform_file(fpath: str) -> float:
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if 'mcsta' in fpath:
                if l.startswith('  Probability: '):
                    return float(l[len('  Probability: '):])
            else:
                if l.startswith('  Estimated probability: '):
                    return float(l[len('  Estimated probability: '):])

        assert False, f"fun get_success_pr_from_modest_file fails to extract success pr from {fpath} "

def get_success_pr_from_modest_file(fpath: str) -> float:
    #print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if 'mcsta' in fpath:
                if l.startswith('  Probability: '):
                    return float(l[len('  Probability: '):])
            else:
                if l.startswith('  Estimated max. probability: '):
                    return float(l[len('  Estimated max. probability: '):])

        assert False, f"fun get_success_pr_from_modest_file fails to extract success pr from {fpath} "

def get_mem_consumption_from_modest_file(fpath: str):
    '''

    :param fpath:
    :return: number of peak bytes used
    '''
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if l.startswith('\ufeffPeak memory usage: '):
                assert l[-3:-1] == 'MB', "Memory consumption must be in MB"
                return float(l[len('\ufeffPeak memory usage: '):-3]) * 2**20 #Return the result in Bytes

        assert False, f"fun get_time_from_modest_file fails to extract simulation memory consumption from {fpath} "

def get_mem_consumption_from_rucop_file(fpath: str):
    '''

    :param fpath:
    :return: number of peak bytes used
    '''
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if l.startswith('Max Memory Consumption: '):
                l = l[len('Max Memory Consumption: '):-1].split(" ")
                assert l[1] == 'bytes', "Memory consumption must be in bytes"
                return int(l[0])

        assert False, f"fun get_time_from_modest_file fails to extract simulation memory consumption from {fpath} "

def get_time_from_modest_file(fpath: str):
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if l.startswith('Simulation time: '):
                assert l[-2] == 's', "Time must be in seconds"
                return float(l[len('Simulation time: '):-2])

        assert False, f"fun get_time_from_modest_file fails to extract simulation time from {fpath} "

def get_time_from_rucop_file(fpath: str):
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if l.startswith('Compute BRUF time: '):
                l = l[len('Compute BRUF time: '):-1].split(" ")
                assert l[1] == 'seconds', "Time must be in seconds"
                return float(l[0])

        assert False, f"fun get_time_from_rucop_file fails to extract computation time from {fpath} "

def get_time_from_lrucop_file(fpath: str):
    print(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            if l.startswith('[Info] IBRUFNFunctionGenerator.generate: takes '):
                l = l[len('[Info] IBRUFNFunctionGenerator.generate: takes '):-1].split(" ")
                assert l[1] == 'seconds.', "Time must be in seconds"
                return float(l[0])

        assert False, f"fun get_time_from_lrucop_file fails to extract computation time from {fpath} "
