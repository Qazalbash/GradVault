def binary_search(lst, item):
    low, high = 0, len(lst) - 1
    while True:
        guess = (low + high) // 2
        if high < low:
            return -1
        elif item == lst[guess]:
            return guess
        elif item < lst[guess]:
            high = guess - 1
        elif item > lst[guess]:
            low = guess + 1


def update_record(student_records, ID, record_title, data):
    if record_title == "ID":
        return "ID cannot be updated"
    record, lst = {
        "Email": 1,
        "Mid1": 2,
        "Mid2": 3
    }, [i[0] for i in student_records]
    if binary_search(lst, ID) == -1:
        return "Record not found"
    if record_title in "Mid1Mid2":
        data = int(data)
    tag = binary_search(lst, ID)
    lst2 = list(student_records[tag])
    lst2[record[record_title]] = data
    student_records[tag] = tuple(lst2)
    return student_records
