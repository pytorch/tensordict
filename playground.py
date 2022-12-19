import os, sys
import torch
from tensor_map_cpp import TensorMap


def main(argv):
    if len(argv) == 1 and argv[0] == '-d':
        print('running in debug mode. Use (lldb) Attach with following pid:')
        print(os.getpid())
        input()

    m = TensorMap()
    x = torch.ones(3)
    m.set('a', x)
    a = m.get('a')
    print('TensorRefCheck:', a is x)

    m2 = TensorMap()
    m.set('b', m2)
    b = m.get('b')
    print('MapRefCheck:', b is m2) # Not working

    m2.set('c', x)
    c = b.get('c')
    print('SubTensorRefCheck:', c is x)
    bc = m.get(('b', 'c'))
    print('PathTensorRefCheck:', bc is c)

    y = torch.rand(3)
    m.set(('x', 'y', 'z'), y)
    xyz = m.get(('x', 'y', 'z'))
    print(xyz)
    print('SetGetPath RefCheck:', xyz is y)

    # Write over existing value test
    v2 = torch.rand(3)
    m2.set(('c', 'd'), v2)
    cd = m2.get(('c', 'd'))
    bcd = m.get(('b', 'c', 'd'))
    print('WriteOverTest RefCheck1:', cd is v2)
    print('WriteOverTest RefCheck2:', bcd is v2)

    # Add more nesting
    m.set(('b', 'c', 'e'), torch.ones(3))
    m.set(('b', 'f'), torch.zeros(3))
    m.set(('x', 'w'), torch.rand(3))
    keys = m.keys()
    print(keys)



if __name__ == "__main__":
    main(sys.argv[1:])
