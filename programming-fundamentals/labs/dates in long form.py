def print_dates_in_long_form(t: dict) -> None:

    month_names = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

    for i in t:
        print(f"{month_names[i['month']]} {i['day']}, {i['year']}")
