"""Quantitative comparison of first- and second-order models."""
import random
from first_order_lm import FirstOrderLM, tokenise as tok1, DATA, START, END
from second_order_lm import SecondOrderLM, tokenise as tok2

train = {" ".join(l.lower().split()) for l in DATA.strip().splitlines()}
m1 = FirstOrderLM().fit(tok1(DATA))
m2 = SecondOrderLM().fit(tok2(DATA))
V = {t for s in tok1(DATA) for t in s}
words = V - {START, END}
n_out = len(V) - 1                     # possible outputs: everything except <START>

# --- parameters
p1 = sum(len(c) for c in m1.counts.values())
p2 = sum(len(c) for c in m2.counts.values())
ctx1, ctx2 = len(m1.counts), len(m2.counts)
full1_simple, full2_simple = len(V) ** 1 * len(V), len(V) ** 2 * len(V)

print(f"Vocabulary size |V| (incl. <START>, <END>): {len(V)}")
print(f"{'':34}{'1st order':>10}{'2nd order':>10}")
print(f"{'observed contexts (CPT rows)':34}{ctx1:>10}{ctx2:>10}")
print(f"{'non-zero parameters (CPT cells)':34}{p1:>10}{p2:>10}")
print(f"{'full table size |V|^k x |V|':34}{full1_simple:>10}{full2_simple:>10}")
print(f"{'fraction of full table observed':34}{p1/full1_simple:>10.3f}{p2/full2_simple:>10.3f}")
print(f"{'zero-prob. contexts (unobserved)':34}{len(V)**1-ctx1:>10}{len(V)**2-ctx2:>10}")

# --- generation diversity
N = 1000
rng = random.Random(0)
for name, m in [("1st order", m1), ("2nd order", m2)]:
    outs = [m.generate("sample", rng=rng) for _ in range(N)]
    sents = [s for s, _ in outs]
    distinct = set(sents)
    novel = {s for s in distinct if s not in train}
    unfinished = sum(1 for _, ok in outs if not ok)
    lens = sum(len(s.split()) for s in sents) / N
    print(f"\n[{name}] {N} sampled sentences: {len(distinct)} distinct, "
          f"{len(novel)} distinct not in training data, "
          f"{unfinished} hit max_len, mean length {lens:.1f}")
    print("   fraction of samples identical to a training sentence:",
          f"{sum(s in train for s in sents)/N:.2f}")
    # grammatical-by-template check: 'the X V on/to the Y' pattern
    ok = sum(1 for s in sents if len(s.split()) == 6 and s.split()[2] in ("sat", "ran")
             and (s.split()[2] == "sat") == (s.split()[3] == "on"))
    print(f"   fraction with well-formed 6-word 'the A V P the B' pattern: {ok/N:.2f}")
    print("   examples of novel sentences:", sorted(novel)[:4])
