def popular_author(b):
    writer = {}
    for i in b:
        writer[i["author"]] = writer.get(i["author"], 0) + 1
    maximum = max(writer.values())
    popularAuthor = [i for i in writer.keys() if writer[i] == maximum]
    return ", ".join(sorted(popularAuthor))
