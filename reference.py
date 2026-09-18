"""Mine short phrasing examples from the optional local reference corpus.

Outputs counts and brief contextual fragments; never adds essays to training.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from common import read_jsonl, write_jsonl
from style import markers, score


def main():
    files = sorted(Path("data/raw").glob("*.jsonl"))
    if not files:
        raise SystemExit("Run collect.py first to download optional style references")
    counts, candidates, seen = {}, defaultdict(list), set()
    for path in files:
        rows = read_jsonl(path)
        counts[path.stem] = {"posts": len(rows), "markers": dict(sum((Counter(markers(r["text"])) for r in rows), Counter()))}
        for row in rows:
            for paragraph in row["text"].split("\n\n"):
                if not score(paragraph):
                    continue
                words = paragraph.split()
                # Keep context small; output is a phrase reference, not a second corpus.
                for term in ("epistemic", "excited", "Do The Thing", "orthogonal", "crux",
                             "load-bearing", "directionally", "operationalise", "operationalize",
                             "legible", "on the margin", "on priors", "cached thought",
                             "object-level", "outside view", "inside view", "steelman",
                             "nontrivial", "counterfactual", "holding fixed", "conditional on",
                             "bottleneck", "sympathetic to", "gears-level", "affordance", "salient"):
                    match = re.search(re.escape(term), paragraph, re.I)
                    if not match:
                        continue
                    start = len(paragraph[:match.start()].split())
                    fragment = " ".join(words[max(0, start - 5):start + 13])
                    key = (term, fragment.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates[term].append({"term": term, "fragment": fragment,
                                             "source": row["source"], "url": row["url"], "author": row["author"]})
    examples = []
    for term, rows in candidates.items():
        # Round-robin across sources instead of taking only the first site's hits.
        sources = defaultdict(list)
        for row in rows:
            sources[row["source"]].append(row)
        for i in range(8):
            for values in sources.values():
                if i < len(values):
                    examples.append(values[i])
    write_jsonl("data/reference/phrases.jsonl", examples)
    Path("data/reference/counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print("Reference only: data/reference/phrases.jsonl; no text added to training")


if __name__ == "__main__":
    main()
