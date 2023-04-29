def sum_digits(num, count = 0):
    if num == 0:
        return count
    count += num % 10
    num = (num - num % 10)//10
    return sum_digits(num, count)
