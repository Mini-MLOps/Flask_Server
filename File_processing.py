import os


class File_processing:
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, content):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def remove(self):
        with open(self.file_path, 'w') as f:
            f.write('')
