def update_record(student_records, ID, record_title, data):
    if record_title == "ID":
        return "ID cannot be updated"
    info, record, flag = {}, {"Email": 0, "Mid1": 1, "Mid2": 2}, 0
    for i in student_records:
        s = [i[1], i[2], i[3]]
        if i[0] == ID:
            if record_title in "Mid1Mid2":
                data = int(data)
            s[record[record_title]], flag = data, 1
        info[i[0]] = s
    if flag == 0:
        return "Record not found"
    return [(key, value[0], value[1], value[2]) for key, value in info.items()]

print(update_record(student_records, ID, record_title, data))
