from typing import Any


def get_length(v: Any) -> int:
    length = 0
    for i in range(len(v)):
        length += v[i]**2
    return length**0.5


def dot_product(v1: Any, v2: Any) -> Any:
    prod = 0
    for i in range(len(v1)):
        prod += v1[i] * v2[i]
    return prod