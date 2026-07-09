import random
from collections import Counter
from typing import Any, Dict, List

from litellm.router_strategy.simple_shuffle import simple_shuffle


class _FakeRouter:
    def print_deployment(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        return deployment


def _dep(dep_id: str, **litellm_params: Any) -> Dict[str, Any]:
    return {
        "model_name": "m",
        "litellm_params": {"model": "gpt-4", **litellm_params},
        "model_info": {"id": dep_id},
    }


def _tally(deployments: List[Dict[str, Any]], samples: int = 20000) -> Counter:
    router = _FakeRouter()
    counts: Counter = Counter()
    for _ in range(samples):
        picked = simple_shuffle(router, [d.copy() for d in deployments], "m")
        counts[picked["model_info"]["id"]] += 1
    return counts


def test_weighted_pick_is_order_independent_and_does_not_starve_unweighted():
    random.seed(1234)
    weighted = _dep("A", weight=9)
    unweighted = _dep("B")

    weighted_first = _tally([weighted, unweighted])
    plain_first = _tally([unweighted, weighted])

    for counts in (weighted_first, plain_first):
        assert counts["B"] > 0, f"un-weighted deployment starved: {counts}"
        share_a = counts["A"] / (counts["A"] + counts["B"])
        assert 0.85 <= share_a <= 0.95, f"weight=9 not honored: {counts}"

    share_a_weighted_first = weighted_first["A"] / sum(weighted_first.values())
    share_a_plain_first = plain_first["A"] / sum(plain_first.values())
    assert abs(share_a_weighted_first - share_a_plain_first) < 0.05
