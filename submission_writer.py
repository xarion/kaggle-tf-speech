class SubmissionWriter:
    def __init__(self, file_name, label_mapping):
        self.file = open(file_name, 'w')
        self.file.write("fname,label\n")
        self.label_mapping = label_mapping

    def add_records(self, predictions, file_names):
        for prediction, file_name in zip(predictions, file_names):
            self.file.write("%s,%s\n" % (file_name, self.label_mapping[prediction]))

    def close(self):
        self.file.flush()
        self.file.close()
