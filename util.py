import math

def _0_if_nan(num: float) -> float:
    return 0.0 if math.isnan(num) else num