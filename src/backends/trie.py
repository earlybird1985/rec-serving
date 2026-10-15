"""Prefix trie and a small beam over fixed-length semantic IDs (A3 §6 / A7 §5.3)."""
from __future__ import annotations


class SemanticIDTrie:
    def __init__(self, semid2item: dict[tuple[int, ...], int]):
        self.root: dict = {}
        for semid in semid2item:
            node = self.root
            for tok in semid:
                node = node.setdefault(int(tok), {})

    def allowed_next(self, prefix: list[int] | tuple[int, ...]) -> list[int]:
        node = self.root
        for tok in prefix:
            node = node.get(int(tok))
            if node is None:
                return []
        return list(node.keys())


def beam_search(
    logits_fn,
    n_levels: int,
    vocab_size: int,
    beam_width: int,
    trie: SemanticIDTrie | None,
    semid2item: dict[tuple[int, ...], int],
) -> tuple[list[tuple[int, float]], float]:
    """Return (item, score) pairs and the fraction of finished beams missing the trie."""
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    for level in range(n_levels):
        expanded: list[tuple[list[int], float]] = []
        for prefix, score in beams:
            logits = list(logits_fn(prefix))
            allowed = set(range(vocab_size))
            if trie is not None:
                allowed = set(trie.allowed_next(prefix))
            for tok, logit in enumerate(logits):
                if tok not in allowed:
                    continue
                expanded.append((prefix + [tok], score + float(logit)))
        expanded.sort(key=lambda pair: (-pair[1], pair[0]))
        beams = expanded[:beam_width]
        if not beams:
            return [], 1.0
    mapped: list[tuple[int, float]] = []
    invalid = 0
    for seq, score in beams:
        item = semid2item.get(tuple(seq), -1)
        if item < 0:
            invalid += 1
        mapped.append((item, score))
    mapped.sort(key=lambda pair: (-pair[1], pair[0] if pair[0] >= 0 else 10**9))
    rate = invalid / len(mapped) if mapped else 1.0
    return mapped, rate
