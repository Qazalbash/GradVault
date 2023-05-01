def sort_rectangles(rectangle_records, record_title):
    for i in range(1, len(rectangle_records)):
        key = rectangle_records[i]
        j = i - 1
        while j >= 0 and key[record_title] < rectangle_records[j][record_title]:
            rectangle_records[j + 1] = rectangle_records[j]
            j -= 1
        rectangle_records[j + 1] = key
    return rectangle_records


print(sort_rectangles(rectangle_records, record_title))
