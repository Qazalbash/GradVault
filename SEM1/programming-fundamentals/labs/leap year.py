def is_leap_year(year):
    if year % 4 == 0:
        if year % 400 != 0 and year % 100 == 0:
            return False
        return True
    return False
