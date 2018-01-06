dataset_labels = ["_background_noise_", "bird", "dog", "eight", "four", "happy", "left", "nine", "off", "one",
                  "seven", "six", "three", "two", "wow", "zero", "bed", "cat", "down", "five", "go", "house",
                  "marvin", "no", "on", "right", "sheila", "stop", "tree", "up", "yes"]

competition_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]

competition_labels_to_ids = dict(zip(competition_labels, range(0, len(competition_labels))))

dataset_labels_to_competition_ids = dict()
dataset_labels_to_competition_labels = dict()
for dataset_label in dataset_labels:
    if dataset_label in competition_labels_to_ids:
        competition_id = competition_labels_to_ids[dataset_label]
    elif dataset_label == "_background_noise_":
        competition_id = competition_labels_to_ids["silence"]
    else:
        competition_id = competition_labels_to_ids["unknown"]
    dataset_labels_to_competition_ids[dataset_label] = competition_id
    competition_label = competition_labels[competition_id]
    dataset_labels_to_competition_labels[dataset_label] = competition_label
