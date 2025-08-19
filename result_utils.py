
def Calculate_BWT(performance_stability):
    bwt_dict = {}
    for k, vals in performance_stability.items():
        if len(vals) < 2:
            continue
        f = vals[0]
        bwt_dict[k] = [v-f for v in vals[1:]]
    
    bwt_values = avg_bwt_per_domain(bwt_dict)
    return bwt_values, bwt_dict
def avg_bwt_per_domain(data: dict)  -> list[float]:
    if not data:
        return []
    
    max_len = max((len(v) for v in data.values()), default=0)
    out = []
    # offset = how far from the end (1 = last element)
    for offset in range(max_len, 0, -1):
        col = [vals[-offset] for vals in data.values() if len(vals) >= offset]
        out.append(sum(col) / len(col))

    return out


def Calculate_FWT(performance_plasticity: dict[str]) -> dict[str, list[float]]:
    fwt_dict = {}
    for k, vals in performance_plasticity.items():
        if len(vals) == 1:
            continue
 
        fwt_dict[k] = vals[0] - vals[1]

    return fwt_dict


def calculate_cost():
    cost = 0.0
    # Implement cost calculation logic here
    return cost


