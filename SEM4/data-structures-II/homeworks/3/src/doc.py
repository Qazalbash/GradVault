import re
from pathlib import Path


class Document:
    def __init__(self, path: str) -> None:

        path: Path = Path(path)
        self.doc_id = str(path)[7:].replace("\\", "/")
        self.word_loc_map = {}
        self.size = 0

        content = path.read_text(encoding="ascii", errors="ignore")
        self.__tokenize_document(re.sub(r"[^\w\s]", "", content).replace("\n", " "))

    def __tokenize_document(self, content: str) -> None:
        start = 0
        while start < len(content):

            if content[start].isalnum():

                stop = content.find(" ", start)
                word = content[start:stop]
                self.word_loc_map[word] = self.word_loc_map.get(word, []) + [
                    (self.doc_id, start, stop)
                ]

                start = stop + 1
                self.size += 1

            else:
                start += 1
