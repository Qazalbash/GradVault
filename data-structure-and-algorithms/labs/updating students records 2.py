def update_record(student_records, ID, record_title, data):
    if record_title == "ID":
        return "ID cannot be updated"
    else:
        signal = 0
        bio = [[i for i in j] for j in student_records]
        for row in range(len(student_records)):
            if bio[row][0] == ID:
                signal = 1
                if record_title == "Email":
                    bio[row][1] = data
                elif record_title == "Mid1":
                    bio[row][2] = int(data)
                elif record_title == "Mid2":
                    bio[row][3] = int(data)
            bio[row] = (bio[row][0], bio[row][1], bio[row][2], bio[row][3])
        if signal == 0:
            return "Record not found"
        return bio
