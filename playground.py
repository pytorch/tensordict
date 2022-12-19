import os, sys
import torch
from tensordict import TensorMap


def main(argv):
    if len(argv) == 1 and argv[0] == '-d':
        print('running in debug mode. Use (lldb) Attach with following pid:')
        print(os.getpid())
        input()

    m = TensorMap()
    x = torch.ones(3)
    m['a'] = x
    a = m['a']
    print('TensorRefCheck:', a is x)

    m2 = TensorMap()
    m['b'] = m2
    b = m['b']
    print('MapRefCheck:', b is m2) # Not working

    m2['c'] = x
    c = b['c']
    print('SubTensorRefCheck:', c is x)
    bc = m['b', 'c']
    print('PathTensorRefCheck:', bc is c)

    y = torch.rand(3)
    m['x', 'y', 'z'] = y
    xyz = m['x', 'y', 'z']
    print(xyz)
    print('SetGetPath RefCheck:', xyz is y)

    # Write over existing value test
    v2 = torch.rand(3)
    m2['c', 'd'] = v2
    cd = m2['c', 'd']
    bcd = m['b', 'c', 'd']
    print('WriteOverTest RefCheck1:', cd is v2)
    print('WriteOverTest RefCheck2:', bcd is v2)

    # Add more nesting
    m['b', 'c', 'e'] = torch.ones(3)
    m['b', 'f'] = torch.zeros(3)
    m['x', 'w'] = torch.rand(3)
    keys = m.keys()
    print(keys)


if __name__ == "__main__":
    main(sys.argv[1:])
