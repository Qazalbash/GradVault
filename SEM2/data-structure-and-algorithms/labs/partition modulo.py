def partition_modulo_n(n, t):
    mydict = {n * i // abs(n): [] for i in range(abs(n))}
    for num in t:
        mydict[num % n] = mydict.get(num % n) + [num]
    return mydict
