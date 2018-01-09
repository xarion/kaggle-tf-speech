import dataset_labels

entries = []
with open("training_list.txt") as f:
    for line in f.readlines():
        entries.append(line.strip())

labeled_entries = dict()
for entry in entries:
    label = entry.split("/")[0]
    competition_label = dataset_labels.dataset_labels_to_competition_labels[label]
    if competition_label not in labeled_entries:
        labeled_entries[competition_label] = []
    labeled_entries[competition_label].append(entry)

label_counts = dict()

for label in labeled_entries.keys():
    labels = dataset_labels.dataset_labels_to_competition_labels
    label_counts[label] = len(labeled_entries[label])

print label_counts

for label in labeled_entries.keys():
    if label != "unknown":
        labeled_entries[label] = labeled_entries[label] * 17

label_counts = dict()

balanced_training_file = open('balanced_training_list.txt', 'w')
for label in labeled_entries.keys():
    labels = dataset_labels.dataset_labels_to_competition_labels
    label_counts[label] = len(labeled_entries[label])
    for item in labeled_entries[label]:
        balanced_training_file.write("%s\n" % item)

print label_counts
