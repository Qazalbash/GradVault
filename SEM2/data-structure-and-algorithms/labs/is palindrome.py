def is_palindrome(s, index=0):
    if index == len(s):
        return True
    elif s[index] == s[-index - 1]:
        return is_palindrome(s, index + 1)
    return False
