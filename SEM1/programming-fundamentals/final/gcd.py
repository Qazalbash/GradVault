def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


a = int(input())
b = int(input())

import inspect

source = inspect.getsource(gcd)
if "for " in source or "while " in source:
    print("Try a recursive approach!")
else:
    result = gcd(a, b)
    if isinstance(result, int):
        print(result)
    else:
        print("Returned value is not an integer.")