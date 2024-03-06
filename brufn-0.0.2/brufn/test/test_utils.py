from brufn.network import Net, Contact

def gen_binomial_net_level_by_ts(levels:int):
    assert levels > 0, "Levels must be greater than 1"
    contacts = []
    ts = 0
    for level in range(1, levels - 1):
        target = int(2 ** level - 1)
        for node in range(int(2 ** (level - 1) - 1), int(2 ** level - 1)):
            contacts.append(Contact(node, target, ts))
            contacts.append(Contact(target, node, ts))
            target += 1
            contacts.append(Contact(node, target, ts+1))
            contacts.append(Contact(target, node, ts + 1))
            target += 1
        ts += 2

    # Link nodes levels - 2 to target node
    for node in range(int(2 ** (levels - 2) - 1), int(2 ** (levels - 1) - 1)):
        contacts.append(Contact(node, 2 ** (levels - 1) - 1, ts))
        contacts.append(Contact(2 ** (levels - 1) - 1, node, ts))
        ts += 1

    return Net(2 ** (levels - 1), contacts)