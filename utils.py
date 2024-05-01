import os
import numpy as np
from typing import Tuple

def get_cifar():
    if not os.path.exists('./cifar-10-batches-py'):
        if not os.path.exists('./cifar-10-python.tar.gz'):
            os.system('wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        os.system('tar -xzvf cifar-10-python.tar.gz')
        # os.system('rm cifar-10-python.tar.gz')

def get_mnist():
    if not os.path.exists('./mnist'):
        os.system('mkdir mnist')
    os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P mnist') if not os.path.exists('./mnist/train-images-idx3-ubyte.gz') else None
    os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P mnist') if not os.path.exists('./mnist/train-labels-idx1-ubyte.gz') else None
    os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P mnist') if not os.path.exists('./mnist/t10k-images-idx3-ubyte.gz') else None
    os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P mnist') if not os.path.exists('./mnist/t10k-labels-idx1-ubyte.gz') else None
    # unzip if the files are not unzipped
    os.system('gunzip mnist/*.gz') if len(os.listdir('./mnist')) < 8 else None


def read_cifar(file_path) -> Tuple[np.ndarray, np.ndarray]:
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle._Unpickler(f)
        data.encoding = 'latin1'
        datadict = data.load()
        X = datadict['data']
        Y = datadict['labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = X.reshape(len(X), 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_cifar(root_path):
    Xs = []
    Ys = []
    for batch in range(1, 6):
        f = os.path.join(root_path, f'data_batch_{batch}')
        X, Y = read_cifar(f)
        Xs.append(X)
        Ys.append(Y)
    X_train = np.concatenate(Xs)
    Y_train = np.concatenate(Ys)
    del X, Y
    X_test, Y_test = read_cifar(os.path.join(root_path, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

def prepare_cifar_images(cifar10_dir, num_training=49000, num_validation=1000):
    X_train, y_train, X_test, y_test = load_cifar(cifar10_dir)
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image

    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_val = X_val.reshape(X_val.shape[0], 3, 32, 32)
    return X_train, y_train, X_val, y_val

def read_mnist(file_path) -> Tuple[np.ndarray, np.ndarray]:
    import struct
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        Y = np.fromfile(f, dtype=np.uint8)
        return X, Y

def load_mnist(root_path):
    X_train, Y_train = read_mnist(os.path.join(root_path, 'train-images-idx3-ubyte'))
    X_test, Y_test = read_mnist(os.path.join(root_path, 't10k-images-idx3-ubyte'))
    return X_train, Y_train, X_test, Y_test

def prepare_mnist_images(mnist_dir, num_training=59000, num_validation=1000):
    X_train, y_train, X_test, y_test = load_mnist(mnist_dir)
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    return X_train, y_train, X_val, y_val

def normalize(image, mean, std):
    if len(image.shape) == 2:  # For grayscale images
        image = image[:, :, np.newaxis]
        mean = mean[0]
        std = std[0]
    elif len(image.shape) == 3 and image.shape[2] != len(mean):  # For RGB images with mismatched channels
        raise ValueError("The number of channels in the image doesn't match the length of mean and std.")
    normalized_image = (image.astype(np.float32) - mean) / std
    return normalized_image


if __name__ == '__main__':
    get_cifar()
    X, Y = read_cifar('./cifar-10-batches-py/data_batch_1')
    # get_mnist()
    # X, Y = read_mnist('./mnist/train-images-idx3-ubyte')
    import cv2
    img = X[1]
    print(img[1])
    label = Y[1]
    print(max(Y))
    print(label)
    img = img.reshape(3, 32, 32).transpose(1, 2, 0)
    # img = img.reshape(28, 28)
    # scale 10 times
    img = cv2.resize(img, (480, 480))
    cv2.imshow('img', img)
    cv2.waitKey(0)