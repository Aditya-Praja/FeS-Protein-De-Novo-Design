import math


def rank(rows, cfg):
    result = []
    for row in rows:
        copy = dict(row)
        reasons = list(row["reasons"])
        score = 0.0
        for name, weight in cfg["weights"].items():
            value = row.get(name)
            if value is None or not math.isfinite(value):
                reasons.append(f"Required ranking measurement unavailable: {name}")
            else:
                score += weight * value
        if cfg["require_network_path"] and not row["network_spans_layers"]:
            reasons.append("No geometrical cofactor path across layers")
        copy.update(reasons=reasons, passed=not reasons, score=score if not reasons else None)
        result.append(copy)
    return sorted(result, key=lambda r: (not r["passed"], -(r["score"] or 0), r["candidate"]))
