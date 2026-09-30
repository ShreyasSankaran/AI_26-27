"""Second-order autoregressive language model (trigram Bayesian network).

Model:  P(Xt | Xt-2, Xt-1)   -- graph  Xt-2 -> Xt <- Xt-1
Sentences are padded with TWO <START> tokens so that the first real word is
predicted from (<START>, <START>) and the second from (<START>, X1).
Estimated purely from counts of observed triples. No ML libraries.
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
    return [[START, START] + line.lower().split() + [END]
            for line in text.strip().splitlines() if line.strip()]


class SecondOrderLM:
    def __init__(self):
        # counts[(w_{t-2}, w_{t-1})][w_t] = C(w_{t-2}, w_{t-1}, w_t)
        self.counts = defaultdict(Counter)

    def fit(self, sentences):
        for s in sentences:
            for a, b, c in zip(s[:-2], s[1:-1], s[2:]):
                self.counts[(a, b)][c] += 1
        return self

    def dist(self, ctx):
        cnt = self.counts.get(tuple(ctx))
        if not cnt:
            return {}
        total = sum(cnt.values())
        return {v: c / total for v, c in cnt.items()}

    def probabilities(self):
        return {ctx: self.dist(ctx) for ctx in self.counts}

    def show(self, ctx):
        d = self.dist(ctx)
        name = f"({ctx[0]}, {ctx[1]})"
        if not d:
            print(f"P(next | {name}) : no observed transitions")
            return
        tot = sum(self.counts[tuple(ctx)].values())
        print(f"P(next | {name}):")
        for v, p in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {v:<8} {self.counts[tuple(ctx)][v]}/{tot} = {p:.4f}")

    def predict(self, ctx):
        d = self.dist(ctx)
        return max(d, key=d.get) if d else None

    def next_token(self, ctx, mode="sample", rng=random):
        d = self.dist(ctx)
        if not d:
            return None
        if mode == "greedy":
            return max(d, key=d.get)
        words, probs = zip(*d.items())
        return rng.choices(words, weights=probs, k=1)[0]

    def generate(self, mode="sample", max_len=20, rng=random):
        out, ctx = [], (START, START)
        while len(out) < max_len:
            nxt = self.next_token(ctx, mode, rng)
            if nxt is None or nxt == END:
                return " ".join(out), True
            out.append(nxt)
            ctx = (ctx[1], nxt)
        return " ".join(out), False

    def check_normalisation(self, tol=1e-9):
        ok = True
        for ctx, d in sorted(self.probabilities().items()):
            total = sum(d.values())
            flag = "OK" if abs(total - 1.0) < tol else "FAIL"
            ok &= flag == "OK"
            print(f"  sum_v P(v | {ctx[0]}, {ctx[1]})".ljust(38) + f"= {total:.6f}  {flag}")
        return ok


if __name__ == "__main__":
    rng = random.Random(42)
    sents = tokenise(DATA)
    lm = SecondOrderLM().fit(sents)

    print("=== Selected conditional probability tables ===")
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("cat", "ran"), ("sat", "on"), ("on", "the"),
                ("ran", "to"), ("to", "the"), ("the", "mat")]:
        lm.show(ctx)
        print()

    print("=== Normalisation test ===")
    print("All rows sum to 1:", lm.check_normalisation(), "\n")

    print("=== Most probable next word ===")
    for ctx in [(START, "the"), ("the", "cat"), ("the", "dog"), ("cat", "sat"),
                ("sat", "on"), ("on", "the"), ("to", "the")]:
        print(f"  argmax P(. | {ctx[0]}, {ctx[1]}) = {lm.predict(ctx)}")
    print()

    print("=== Unseen context ===")
    print("  P(next | ('cat','dog')) =", lm.dist(("cat", "dog")), "-> generation stops\n")

    print("=== 20 sampled sentences ===")
    gen = [lm.generate("sample", rng=rng)[0] for _ in range(20)]
    for i, s in enumerate(gen, 1):
        print(f"  {i:2d}. {s}")
    with open("generated_second_order.txt", "w") as f:
        f.write("\n".join(gen) + "\n")

    print("\n=== Greedy vs sampling (5 sentences each) ===")
    print(" Greedy:")
    for _ in range(5):
        s, done = lm.generate("greedy")
        print("   ", s, "" if done else "   [hit max_len]")
    print(" Sampling:")
    for _ in range(5):
        print("   ", lm.generate("sample", rng=rng)[0])
