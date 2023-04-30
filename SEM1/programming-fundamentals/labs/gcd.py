# This peice of code will calculate the gcd of two number
# a and b for you recursively

def gcd(a,b):

    # if b == 0 then it will return a
    
    if b == 0:
        return a
        
    # else it will take the remainder of a divide by b
    # and swap the position of b with it
    # the purpose to do so is to meet our base case
    
    return gcd(b, a%b)
