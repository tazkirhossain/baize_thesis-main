"""A shared, stateless domain role prevents cross-client evaluation leakage."""
import hashlib
import math


def shared_domain_split(domain: str, seed: int, fractions=(0.70, 0.15, 0.15)) -> int:
    """Return train=0, val=1, test=2 using the same rule on every client.

    No labels, client IDs, row counts or evaluation metrics enter this rule.
    Fractions are probability targets over domains, not exact row quotas.
    Do not search for a different seed after inspecting evaluation performance.
    """
    if (not isinstance(domain, str) or not domain or len(fractions) != 3 or
            any(not math.isfinite(f) or f <= 0 for f in fractions) or
            not math.isclose(sum(fractions), 1.0, abs_tol=1e-9)):
        raise ValueError("invalid shared domain split configuration")
    value = int.from_bytes(hashlib.sha256(
        f"baize-domain-role-v1:{seed}:{domain}".encode("utf-8")).digest(), "big")
    space = 1 << 256
    if value < int(fractions[0] * space):
        return 0
    if value < int((fractions[0] + fractions[1]) * space):
        return 1
    return 2
