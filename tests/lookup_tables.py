import tensorflow as tf

with tf.Session() as sess:
    dataset_labels = ["_background_noise_", "bird", "dog", "eight", "four", "happy", "left", "nine", "off", "one",
                      "seven", "six", "three", "two", "wow", "zero", "bed", "cat", "down", "five", "go", "house",
                      "marvin", "no", "on", "right", "sheila", "stop", "tree", "up", "yes"]

    competition_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence",
                          "unknown"]

    competition_labels_to_ids = dict(zip(competition_labels, range(0, len(competition_labels))))

    dataset_labels_to_ids = dict()
    for dataset_label in dataset_labels:
        if dataset_label in competition_labels_to_ids:
            dataset_labels_to_ids[dataset_label] = competition_labels_to_ids[dataset_label]
        elif dataset_label == "_background_noise_":
            dataset_labels_to_ids[dataset_label] = competition_labels_to_ids["silence"]
        else:
            dataset_labels_to_ids[dataset_label] = competition_labels_to_ids["unknown"]

    items = dataset_labels_to_ids.items()

    keys, values = zip(*items)

    input_tensor = tf.convert_to_tensor(dataset_labels)

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
    )
    out = table.lookup(input_tensor)
    sess.run(table.init)
    sess.run(tf.global_variables_initializer())
    output, = sess.run([out])
    count = 0
    for o in output:
        print((dataset_labels[count], competition_labels[o]))
        count += 1
