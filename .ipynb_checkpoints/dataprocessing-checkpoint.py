def load_keyPoints(flatten=False, scale=False):
    dataset, labels, label_track = [], [], 0
    parent_dir = "/Users/calvin/Documents/NUIG/Thesis/gesture-estimator/extractedData/"
    actions = ['handshake', 'waving', 'yawning', 'walking', 'bowing', 'punching', 'standing', 'sitting', 'touchinghead', 'defending', 'reachingup']
    for action in actions:
        label_track += 1
        files = os.listdir(os.path.join(parent_dir, action))
        files.remove(".DS_Store") if ".DS_Store" in files else files
        for video_file in files:
            video_file = np.load(os.path.join(parent_dir, action, video_file))
            flattened_video = []
            for frame in video_file:
                if flatten:
                    frame = frame.reshape(-1)
                if scale:
                    frame[:, [4, 8, 12, 16, 20]] = standardize(frame[:, [4, 8, 12, 16, 20]])
                    frame[:, [5, 9, 13, 17, 21]] = standardize(frame[:, [5, 9, 13, 17, 21]])
                    frame[:, [6, 10, 14, 18]]    = standardize(frame[:, [6, 10, 14, 18]])
                    frame[:,  7] = standardize(frame[:,  7])
                    frame[:, 11] = standardize(frame[:, 11])
                    frame[:, 15] = standardize(frame[:, 15])
                    frame[:, 19] = standardize(frame[:, 19])
                flattened_video.append(frame)
            flattened_video = np.array(flattened_video)
            dataset.append(np.array(flattened_video))
            labels.append(label_track)
    return dataset, labels, frame


def standardize(frame):
    mean = np.mean(frame, axis=0)
    std = np.std(frame, axis=0)+0.000001
    frame = (frame - mean) / std
    return frame


def frame_interpolation(video, limit):
    output_video, count = [], 0
    if video.shape[0] < limit:
        while count < limit - video.shape[0]:
            output_video.append(video[count])
            output_video.append(video[count])
            count += 1
        while count < video.shape[0]:
            output_video.append(video[count])
            count += 1
    else:
        while count < 2* (video.shape[0] - limit):
            if count % 2 == 0:
                output_video.append(video[count])
            count += 1
        while count < video.shape[0]:
            output_video.append(video[count])
            count += 1
    return np.array(output_video)


def flatten_dataset(dataset, level=1):
    reshapped_dataset = []
    for video in dataset:
        if level == 1:
            reshapped_dataset.append(video.reshape(216, 748))
        elif level == 2:
            reshapped_dataset.append(video.reshape(1, 161568))
    return reshapped_dataset


def encode_labels(labels):
    encoded_labels = []
    for label in labels:
        tmp = np.zeros(11)
        tmp[label-1] = 1
        encoded_labels.append(tmp)
    return encoded_labels


def shuffle_dataset(dataset, labels):
    shuffledDataset = list(zip(dataset, labels))
    random.shuffle(shuffledDataset)
    return shuffledDataset


def train_test_split(shuffledDataset, train_test_ratio = 0.8, flatten_label=False):
    train_test_ratio = round(len(shuffledDataset) * train_test_ratio)
    train, train_label = zip(*shuffledDataset[:train_test_ratio])
    test , test_label  = zip(*shuffledDataset[train_test_ratio:])

    assert len(train) + len(test) == len(shuffledDataset)
    
    train = np.array(train)
    train_label = np.array(train_label)
    
    test = np.array(test)
    test_label = np.array(test_label)
    
    if not flatten_label:
        train_label = train_label.reshape(train.shape[0], 1, 11)
        test_label = test_label.reshape(test.shape[0], 1, 11)
    
    print("Train dataset shape :", train.shape)
    print("Test dataset shape  :", test.shape)
    print("Train label shape   :", train_label.shape)
    print("Test label shape    :", test_label.shape)
    
    return train, train_label, test, test_label