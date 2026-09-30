"""First-order autoregressive language model (bigram Bayesian network).

Model:  X1 -> X2 -> ... -> XT   with   P(Xt | Xt-1)
Estimated purely from transition counts. No ML libraries.
"""
import random
from collections import defaultdict, Counter

START, END = "<START>", "<END>"

DATA = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""


def tokenise(text):
    """Lower-case, split on whitespace, add <START>/<END>."""
    return [[START] + line.lower().split() + [END]
            for line in text.strip().splitlines() if line.strip()]


class FirstOrderLM:
    def __init__(self):
        # counts[prev][next] = C(prev, next)   <-- transition counts live here
        self.counts = defaultdict(Counter)

    def fit(self, sentences):
        for sent in sentences:
            for prev, nxt in zip(sent[:-1], sent[1:]):
                self.counts[prev][nxt] += 1
        return self

    # P(Xt | Xt-1) = C(w_i, w_j) / sum_k C(w_i, w_k)   <-- computed here
    def probabilities(self):
        return {w: {v: c / sum(cnt.values()) for v, c in cnt.items()}
                for w, cnt in self.counts.items()}

    def dist(self, prev):
        cnt = self.counts.get(prev)
        if not cnt:
            return {}                      # unseen context -> empty distribution
        total = sum(cnt.values())
        return {v: c / total for v, c in cnt.items()}

    def show(self, prev):
        d = self.dist(prev)
        if not d:
            print(f"P(next | {prev}) : no observed transitions")
            return
        print(f"P(next | {prev}):")
        for v, p in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {v:<8} {self.counts[prev][v]}/{sum(self.counts[prev].values())} = {p:.4f}")

    def predict(self, prev):
        """arg max_w P(w | prev). Ties broken by first-observed order."""
        d = self.dist(prev)
        return max(d, key=d.get) if d else None

    def next_token(self, prev, mode="sample", rng=random):
        d = self.dist(prev)
        if not d:
            return None
        if mode == "greedy":
            return max(d, key=d.get)
        words, probs = zip(*d.items())
        return rng.choices(words, weights=probs, k=1)[0]

    def generate(self, mode="sample", max_len=20, rng=random):
        """X1 ~ P(X1|<START>), X2 ~ P(X2|X1), ... stop at <END> (or max_len)."""
        out, prev = [], START
        while len(out) < max_len:
            nxt = self.next_token(prev, mode, rng)
            if nxt is None or nxt == END:
                return " ".join(out), True       # terminated properly
            out.append(nxt)
            prev = nxt
        return " ".join(out), False              # hit max_len (did not reach <END>)

    def check_normalisation(self, tol=1e-9):
        ok = True
        for w, d in sorted(self.probabilities().items()):
            total = sum(d.values())
            flag = "OK" if abs(total - 1.0) < tol else "FAIL"
            ok &= flag == "OK"
            print(f"  sum_v P(v | {w:<8}) = {total:.6f}  {flag}")
        return ok


if __name__ == "__main__":
    rng = random.Random(42)
    sents = tokenise(DATA)
    lm = FirstOrderLM().fit(sents)

    print("=== Conditional probability tables ===")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        lm.show(w)
        print()

    print("=== Normalisation test ===")
    print("All rows sum to 1:", lm.check_normalisation(), "\n")

    print("=== Zero-probability transitions (among the five requested words) ===")
    vocab = sorted({t for s in sents for t in s} - {START})
    for w in ["the", "cat", "dog", "sat", "ran"]:
        zeros = [v for v in vocab if lm.dist(w).get(v, 0) == 0]
        print(f"  {w:<4} -> {zeros}")
    print()

    print("=== Most probable next word ===")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to"]:
        print(f"  argmax P(. | {w:<8}) = {lm.predict(w)}")
    print()

    print("=== Unseen context ===")
    print("  P(next | 'elephant') =", lm.dist("elephant"), "-> generation stops")
    print()

    print("=== 20 sampled sentences ===")
    gen = [lm.generate("sample", rng=rng)[0] for _ in range(20)]
    for i, s in enumerate(gen, 1):
        print(f"  {i:2d}. {s}")
    with open("generated_first_order.txt", "w") as f:
        f.write("\n".join(gen) + "\n")

    print("\n=== Greedy vs sampling (5 sentences each) ===")
    print(" Greedy:")
    for _ in range(5):
        s, done = lm.generate("greedy")
        print("   ", s, "" if done else "   [hit max_len, never reached <END>]")
    print(" Sampling:")
    for _ in range(5):
        print("   ", lm.generate("sample", rng=rng)[0])
