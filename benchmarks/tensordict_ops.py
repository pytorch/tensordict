import timeit

import torch
from tensordict import TensorDict

if __name__ == "__main__":
    # check that a str key is faster to gather than nested
    print("\n\n\nKey membership")
    b = torch.randn(3)
    a = torch.rand(3)
    a2 = {"a": a.clone()}
    a1 = {"a": a2}
    dicto = {"a": a1, "b": b.clone()}
    td = TensorDict(dicto, [])
    print("str in dict", timeit.timeit("'b' in dicto", globals={"dicto": dicto}))
    print("str in nested", timeit.timeit("'b' in td.keys(True)", globals={"td": td}))
    print(
        "nested1 in nested",
        timeit.timeit("('a', ) in td.keys(True)", globals={"td": td}),
    )
    print(
        "nested2 in nested",
        timeit.timeit("('a', 'a', ) in td.keys(True)", globals={"td": td}),
    )
    print(
        "nested3 in nested",
        timeit.timeit("('a', 'a', 'a') in td.keys(True)", globals={"td": td}),
    )
    print("basic", timeit.timeit("'b' in td.keys()", globals={"td": td}))

    print("\n\n\nGet")
    print("dict[key]", timeit.timeit("dicto['b']", globals={"dicto": dicto}))
    print("nested[nested1]", timeit.timeit("td.get(('a',))", globals={"td": td}))
    print("nested[nested2]", timeit.timeit("td.get(('a','a'))", globals={"td": td}))
    print("nested[nested3]", timeit.timeit("td.get(('a','a','a'))", globals={"td": td}))
    print("basic", timeit.timeit("td.get('b')", globals={"td": td}))

    # this one is a bit misleading as we check that the value is present and if it is and it is identical
    # we do nothing
    print("\n\n\nset")
    print(
        "dict[key]", timeit.timeit("dicto['b'] = b", globals={"dicto": dicto, "b": b})
    )
    print("basic", timeit.timeit("td.set('b', b)", globals={"td": td, "b": b}))
    print(
        "nested[nested1]",
        timeit.timeit("td.set(('a',), a1)", globals={"td": td, "a1": a1}),
    )
    print(
        "nested[nested2]",
        timeit.timeit("td.set(('a','a'), a2)", globals={"td": td, "a2": a2}),
    )
    print(
        "nested[nested3]",
        timeit.timeit("td.set(('a','a','a'), a)", globals={"td": td, "a": a}),
    )
