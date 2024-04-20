import os


class FileHelper:
    def removeAll(self, folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def insertOne(self, path, file_name, file):
        with open("{}/{}".format(path, file_name), "w") as f:
            f.write(file)

    def findFile(self,folder_path,file_name):
            file_path = os.path.join(folder_path, file_name)
            return os.path.exists(file_path)
