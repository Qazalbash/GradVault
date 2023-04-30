def print_by_anagrams(t):

    ana = {}
    for word in t:
        k = "".join(sorted(word))
        ana[k] = ana.get(k, "") + word + " "
        
    for v in ana.values():
        v = sorted(v.split(" "))
        v = " ".join(v)
        print(v[1:])
    
t = eval(input())
print_by_anagrams(t)
