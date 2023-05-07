def is_leap_year(year):
    if year % 4 == 0:
        return not (year % 400 != 0 and year % 100 == 0)
    return False
