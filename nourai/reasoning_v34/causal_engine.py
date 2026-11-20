"""JABRIX AI v3.4.0 — Causal Relationship Analysis & Cause-and-Effect Graph Construction."""
from __future__ import annotations

from collections import deque
import re
from typing import Any

# Multilingual causal patterns: (regex, cause_group, effect_group, relation_label)
_CAUSAL_PATTERNS: tuple[tuple[re.Pattern[str], int, int, str], ...] = (
    # English (allows decimal dots like 2.3x inside cause/effect phrases)
    (re.compile(r"((?:[^.;\n]|\.(?=\d)){3,80}?)\s+(?:causes|leads to|results in|triggers|produces)\s+((?:[^.;\n]|\.(?=\d)){3,80})", re.I), 1, 2, "causes"),
    (re.compile(r"((?:[^.;\n]|\.(?=\d)){3,80}?)\s+(?:because of|due to|caused by)\s+((?:[^.;\n]|\.(?=\d)){3,80})", re.I), 2, 1, "causes"),
    (re.compile(r"((?:[^.;\n]|\.(?=\d)){3,80}?)\s+because\s+((?:[^.;\n]|\.(?=\d)){3,80})", re.I), 2, 1, "causes"),
    # Arabic
    (re.compile(r"([^\n.;،]{3,80}?)\s+(?:يؤدي إلى|يسبب|ينتج عنه)\s+([^\n.;،]{3,80})"), 1, 2, "causes"),
    (re.compile(r"([^\n.;،]{3,80}?)\s+(?:بسبب|لأن)\s+([^\n.;،]{3,80})"), 2, 1, "causes"),
    # French
    (re.compile(r"((?:[^.;\n]|\.(?=\d)){3,80}?)\s+(?:entraîne|provoque|cause)\s+((?:[^.;\n]|\.(?=\d)){3,80})", re.I), 1, 2, "causes"),
    (re.compile(r"((?:[^.;\n]|\.(?=\d)){3,80}?)\s+(?:parce que|car|en raison de)\s+((?:[^.;\n]|\.(?=\d)){3,80})", re.I), 2, 1, "causes"),
)


def _clean_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip(" ,.;:!?\"'()[]"))


class CausalEngine:
    """Extracts cause-and-effect edges and constructs an analyzable causal graph."""

    def analyze(self, texts: list[str]) -> dict[str, Any]:
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()
        adj: dict[str, list[str]] = {}
        indeg: dict[str, int] = {}
        nodes: set[str] = set()

        for src_idx, text in enumerate(texts):
            for pat, c_grp, e_grp, rel in _CAUSAL_PATTERNS:
                for m in pat.finditer(text or ""):
                    cause = _clean_phrase(m.group(c_grp))
                    effect = _clean_phrase(m.group(e_grp))
                    if len(cause) < 3 or len(effect) < 3 or cause.lower() == effect.lower():
                        continue
                    key = (cause.lower(), effect.lower())
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    nodes.add(cause)
                    nodes.add(effect)
                    adj.setdefault(cause, []).append(effect)
                    indeg.setdefault(cause, 0)
                    indeg[effect] = indeg.get(effect, 0) + 1
                    edges.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "relation": rel,
                            "source_index": src_idx,
                        }
                    )

        root_causes = sorted(n for n in nodes if indeg.get(n, 0) == 0)
        terminal_effects = sorted(n for n in nodes if not adj.get(n))

        # Extract causal chains via bounded BFS from root causes (prevents cycle loops)
        chains: list[list[str]] = []
        for root in root_causes[:10]:
            q: deque[list[str]] = deque([[root]])
            while q and len(chains) < 20:
                path = q.popleft()
                last = path[-1]
                nxt_nodes = adj.get(last, [])
                if not nxt_nodes or len(path) >= 6:
                    if len(path) >= 2:
                        chains.append(path)
                    continue
                for nxt in nxt_nodes:
                    if nxt not in path:
                        q.append(path + [nxt])

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": sorted(nodes),
            "edges": edges,
            "root_causes": root_causes,
            "terminal_effects": terminal_effects,
            "causal_chains": chains,
        }
