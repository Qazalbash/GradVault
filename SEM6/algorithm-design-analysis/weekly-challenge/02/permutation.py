from math import sqrt


def permute(m: int, s: int, a: list[int]):
    ''' 
    Returns YES if you can append several elements to the array
    a, that their sum equals S and the resultant array is a permutation. Returns NO 
    otherwise.

    Parameters:
    - m : Length of array a
    - s : Sum of missing numbers of permutation
    - a : Known numbers of permutation
    
    Constraints:
    - 1 <= m <= 100
    - 1 <= s <= 10000
    - 1 <= a_i <= 100
    '''
    if not (1 <= m <= 100 and 1 <= s <= 10000 and
            all(1 <= i <= 100 for i in a)):
        return "NO"
    # Sum of known numbers
    S = sum(a)
    # Sum of known numbers + sum of missing numbers = sum of permutation
    t = s + S
    # Solving for the roots of the quadratic equation n^2 + n - 2t = 0
    n = (sqrt(1 + 8 * t) - 1) // 2
    # Sum of the first n natural numbers
    S_ = (n * n + n) / 2
    # If the sum of the known numbers equals the sum of the permutation
    if t == S_:
        return "YES"
    return "NO"
