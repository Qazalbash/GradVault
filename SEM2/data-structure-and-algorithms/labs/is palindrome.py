def is_palindrome(s, index = 0):
    if index == len(s):
        return True
    elif s[index] == s[-index-1]:
        return is_palindrome(s, index + 1)
    return False

# I, flag = 0, True
# def is_palindrome(s):
#     global I, flag
#     try:
#         if s[I] == s[-I-1]:
#             I += 1
#             return is_palindrome(s)
#         else:
#             flag = False
#             return flag
#     except:
#         return flag