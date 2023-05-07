def days_conversion(no_of_days: int) -> None:
    years = no_of_days // 365
    months = (no_of_days % 365) // 30
    weeks = ((no_of_days % 365) % 30) // 7
    days = no_of_days - 7 * weeks - 30 * months - 365 * years
    print("Years:", years)
    print("Months:", months)
    print("Weeks:", weeks)
    print("Days:", days)
