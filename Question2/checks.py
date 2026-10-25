"""Table 1: sanity checks of the transformation and window functions (synthetic tensors only)."""
import json, os
import torch
from mm import mirror, rot, g_apply, g_inverse, window

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1

rows = []


def add(name, expected, yours):
    rows.append({"check": name, "expected": expected, "yours": yours})


def maxabs_and_equal(a, b):
    return f"{(a - b).abs().max().item():g}, torch.equal={torch.equal(a, b)}"


add("(a) max|M(M(x)) - x|", "0", maxabs_and_equal(mirror(mirror(x)), x))
add("(b) max|R^4(x) - x|", "0", maxabs_and_equal(rot(x, 4), x))
add("(c) max|M R M(x) - R^-1(x)|", "0", maxabs_and_equal(mirror(rot(mirror(x), 1)), rot(x, -1)))


def bright(v):
    idx = (v == v.max()).nonzero()[0]
    return f"({idx[0].item()}, {idx[2].item()}, {idx[3].item()})"


add("(d) bright pixel of M(p) (frame, row, column)", "(7, 10, 43)", bright(mirror(p)))
add("(e) bright pixel of R(p) (frame, row, column)", "(7, 43, 10)", bright(rot(p, 1)))

# frame f is filled with the value f (frames numbered 1..20)
v = torch.stack([torch.full((1, 64, 64), float(f)) for f in range(1, 21)])
for k, name, exp in [(1, "(f) 1x input / target frames", "1-5 / 6-10"),
                     (2, "(g) 2x input / target frames", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19"),
                     (3, "(h) 3x input / target frames", "1, 4, 7, 10, 13 / 16, 19")]:
    X, Y, _, _ = window(v, 1, k)
    xs = [int(a[0, 0, 0].item()) for a in X]
    ys = [int(a[0, 0, 0].item()) for a in Y]
    if k == 1:
        s = f"{xs[0]}-{xs[-1]} / {ys[0]}-{ys[-1]}" if xs == list(range(xs[0], xs[-1] + 1)) else f"{xs} / {ys}"
    else:
        s = f"{', '.join(map(str, xs))} / {', '.join(map(str, ys))}"
    add(name, exp, s)

# extra: all eight group elements are distinct and g^-1 undoes g
xs = torch.rand(2, 5, 1, 64, 64)
inv_ok = all(torch.equal(g_inverse(g_apply(xs, g), g), xs) for g in range(8))
distinct = len({tuple(g_apply(torch.arange(64 * 64.).view(1, 1, 1, 64, 64), g).flatten()[:6].tolist()) for g in range(8)})
rows.append({"check": "(extra) g^-1(g(x)) == x for all 8 g in G; number of distinct group elements",
             "expected": "True; 8", "yours": f"{inv_ok}; {distinct}"})

json.dump(rows, open(os.path.join(HERE, "results", "table1.json"), "w"), indent=1)
for r in rows:
    print(f"{r['check']:<62} expected {r['expected']:<38} yours {r['yours']}")
