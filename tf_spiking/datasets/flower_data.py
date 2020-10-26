import numpy as np
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import re
import matplotlib.pyplot as plt


def download(target):
    print("downloading flower dataset...")
    url = urlopen("https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip")
    data = url.read()
    file = BytesIO(data)
    zf = ZipFile(file)

    # get the list of files in the zip file
    zip_namelist = zf.namelist()

    data = []
    for class_id in range(1, 102 + 1):
        data.append([class_id, len([name for name in zip_namelist if re.match(f".*/{class_id}/image_.*\.jpg", name)])])
    data = np.array(data)
    i = np.argsort(data[:, -1])
    # print(data[i])
    # take the 10 classes with the most images
    classes = data[i, 0][-10:]

    num_classes = 10  # len(classes)
    target_width, target_height = [500, 400]


    filenames = []
    y = []
    for index, class_id in enumerate(classes):
        # get all the files of a class (train and valid)
        files = [name for name in zip_namelist if re.match(f".*/{class_id}/image_.*\.jpg", name)]
        print(files)
        # take only the first 120 images
        files = files[:120]
        # add the class indices
        y += [[index] * len(files)]
        # and the filenames
        filenames += [files]

    y = np.array(y).astype(np.uint8)
    max_count = len(filenames)
    data = np.zeros([len(filenames[0]), max_count, target_height, target_width, 3], dtype=np.uint8)
    for i in range(max_count):
        for j in range(len(filenames[i])):
            file = BytesIO(zf.open(filenames[i][j]).read())
            im = plt.imread(file, format="jpg")
            h, w, c = im.shape
            data[j, i, :] = im[(h - target_height) // 2:(h - target_height) // 2 + target_height,
                            (w - target_width) // 2:(w - target_width) // 2 + target_width]

    zf.close()

    all_ = {"x": data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1]), "y": y.T.reshape(-1)}

    np.save(target, all_)


def split_data(xy, ratio):
    x, y = xy
    N = x.shape[0]
    N_train = int(np.ceil(N * ratio))
    x_train = x[N_train:]
    y_train = y[N_train:]

    x_test = x[:N_train]
    y_test = y[:N_train]
    return (x_train, y_train), (x_test, y_test)


def load_data(split=0.166666666666):
    cached_file = Path(__file__).parent / "flower_data.npy"
    if not cached_file.exists():
        download(cached_file)

    print("loading data...")
    data = np.load(cached_file, allow_pickle=True)[()]
    x, y = data["x"].astype(np.float32)/255, to_categorical(data["y"])

    return split_data((x,y), split)
