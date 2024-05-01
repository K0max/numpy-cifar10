import numpy as np
from utils import *
from typing import List
import pickle

# all input(x) and output(y) and biases are column vectors
# to consist with the math convention

# x means input(x_size * 1), y means output(y_size * 1), z means ultimate output of network after softmax(z_size * 1)
# cause input is a keywork in python, so I use x instead of input

class fc_layer:
    class relu:
        def forward(x) -> np.ndarray:
            return np.maximum(0, x)
        
        def backward(x) -> np.ndarray:
            return np.where(x > 0, 1, 0)

    class identical_map:
        def forward(x):
            return x
        
        def backward(x):
            return x
    def __init__(self, x_size, y_size, activation=relu, std=1e-2):
        # W is a matrix, shape: (x_size, y_size), so I use capital W
        self.W = np.random.randn(x_size, y_size) * std
        # self.W = np.random.rand(x_size, y_size) * std
        # b is a column vector, shape: (y_size, 1)
        self.b = np.random.randn(y_size, 1) * std
        # activation function, only provide relu now, softmax is implemented in the network
        self.activation = activation
        self.num_neurons = y_size
        if activation == None:
            # identical activation function
            self.activation = fc_layer.identical_map

    # x_shape = (x_size, 1) -> (y_size, 1)
    def forward(self, x) -> np.ndarray:
        # convert to column vector (x_size, 1)
        self.x = x.reshape(-1, 1)
        self.y = self.activation.forward(np.dot(self.W.T, self.x) + self.b)
        return self.y

    def backward(self, learning_rate, grad: np.ndarray, reg=0.5) -> np.ndarray:
        grad = self.activation.backward(grad)
        delta_W = np.dot(self.x, grad.T)
        delta_b = np.sum(grad, axis=1, keepdims=True)
        non_updated_W = self.W
        self.W -= learning_rate * delta_W
        self.b -= learning_rate * delta_b
        return np.dot(non_updated_W, grad)


class denseNet:
    # cause cross_entropy is to compute ultimate output, now z is y
    class cross_entropy:
        # y_hat shape: (z_size,1) nad is a probability distribution and also can be represented as p_i
        def forward(y_hat: np.ndarray, one_hot_y: np.ndarray) -> np.ndarray:
            epsilon = 1e-12
            return -np.sum(one_hot_y * np.log(y_hat + epsilon))
        
        def backward(y_hat: np.ndarray, one_hot_y: np.ndarray) -> np.ndarray:
            return y_hat - one_hot_y

    def __init__(self, layers: List[fc_layer]):
        self.layers = layers
        self.num_layers = len(layers)

    def forward(self, x) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        # softmax x to get probability distribution
        z = np.exp(x - np.max(x))
        z = z / np.sum(z)
        return z

    def backward(self, grad, learning_rate) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(learning_rate, grad)

    def SGD_optimize(self, learning_rate: float, batch: np.ndarray, loss=cross_entropy):
        # batch = np.random.shuffle(batch)
        np.random.shuffle(batch)
        for i, (x_vec, label) in enumerate(batch):
            # y_hat is a probability distribution, shape: (num_classes, 1)
            y_hat = self.forward(x_vec)
            # convert label to one-hot vector to compute loss
            one_hot_y = np.zeros((self.layers[-1].num_neurons, 1))
            one_hot_y[label, 0] = 1
            # print(one_hot_y, y_hat)
            # loss_val = loss.forward(y_hat, one_hot_y)
            # compute gradient of loss
            grad = loss.backward(y_hat, one_hot_y)
            # backpropagate
            self.backward(grad, learning_rate)

    def BGD_optimize(self, learning_rate: float, batch: np.ndarray, loss=cross_entropy):
        batch_outputs = []
        batch_one_hot_y = []
        for instance in batch:
            batch_outputs.append(self.forward(instance[0]))
            one_hot_y = np.zeros((self.layers[-1].num_neurons, 1))
            batch_one_hot_y.append(one_hot_y)
        batch_outputs = np.array(batch_outputs)
        loss_val = 0
        for i in range(len(batch)):
            loss_val += loss.forward(batch_outputs[i], batch_one_hot_y[i])
        loss_val /= len(batch)
        grad = loss.backward(batch_outputs, batch_one_hot_y)
        self.backward(grad, learning_rate)

    def MBGD_optimize(self, learning_rate: float, batch: np.ndarray, loss=cross_entropy):
        mini_batch_size = len(batch) / 10
        for mini_index in range(0, len(batch), mini_batch_size):
            mini_batch = batch[mini_index: mini_index + mini_batch_size]
            mini_batch_outputs = []
            mini_batch_one_hot_y = []
            for instance in mini_batch:
                mini_batch_outputs.append(self.forward(instance[0]))
                one_hot_y = np.zeros((self.layers[-1].num_neurons, 1))
                mini_batch_one_hot_y.append(one_hot_y)
            mini_batch_outputs = np.array(mini_batch_outputs)
            loss_val = 0
            for i in range(mini_batch_size):
                loss_val += loss.forward(mini_batch_outputs[i], mini_batch_one_hot_y[i])
            loss /= mini_batch_size
            grad = loss.backward(mini_batch_outputs, mini_batch_one_hot_y)
            self.backward(grad, learning_rate)

    # batch_size specified in main.py
    def train(self, learning_rate: float, num_epoch: int, batch_size: int, dataset_root_path: str,
              optimizer="SGD", loss=cross_entropy):
        # read data from dataset_root_path
        if "cifar" in dataset_root_path:
            X_train, Y_train, X_test, Y_test = load_cifar(dataset_root_path)
        elif "mnist" in dataset_root_path:
            X_train, Y_train, X_test, Y_test = load_mnist(dataset_root_path)
        train_set = [(X_train[i], Y_train[i]) for i in range(len(X_train))]
        test_set = [(X_test[i], Y_test[i]) for i in range(len(X_test))]
        # normalize
        # train_set = [(x / 255, y) for x, y in train_set]
        # test_set = [(x / 255, y) for x, y in test_set]
        for epoch_index in range(num_epoch):
            for batch_index in range(0, len(train_set), batch_size):
                batch = train_set[batch_index * batch_size: (batch_index + 1) * batch_size]
                if optimizer == "SGD":
                    self.SGD_optimize(learning_rate, batch, loss)
                elif optimizer == "BGD":
                    self.BGD_optimize(learning_rate, batch, loss)
                elif optimizer == "MBGD":
                    self.MBGD_optimize(learning_rate, batch, loss)
            # compute loss and accuracy
            loss_val = 0
            correct_num = 0
            # randomly choose some instances to test
            test_data_idx = np.random.choice(len(test_set), 1000, replace=False)
            test_data = [test_set[i] for i in test_data_idx]
            for instance in test_data:
                y_hat = self.forward(instance[0])
                one_hot_y = np.zeros((self.layers[-1].num_neurons, 1))
                one_hot_y[instance[1], 0] = 1
                loss_val += loss.forward(y_hat, one_hot_y)
                if np.argmax(y_hat) == instance[1]:
                    # if epoch_index == num_epoch - 1:
                    #     print(y_hat, instance[1])
                    correct_num += 1
            loss_val /= len(test_data)
            accuracy = correct_num / len(test_data)
            print(f"Epoch {epoch_index+1}, loss_val: {loss_val:.3f}, accuracy: {accuracy}")
        # validation
        correct_num = 0
        validation_idx = np.random.choice(len(train_set), 1000, replace=False)
        validation_set = [train_set[i] for i in validation_idx]
        for instance in validation_set:
            y_hat = self.forward(instance[0])
            if np.argmax(y_hat) == instance[1]:
                correct_num += 1
        accuracy = correct_num / len(validation_set)
        print(f"Validation accuracy: {accuracy}")

        print("Training finished.")

    def test(self, dataset_root_path: str):
        if "cifar" in dataset_root_path:
            X_train, Y_train, X_test, Y_test = load_cifar(dataset_root_path)
        elif "mnist" in dataset_root_path:
            X_train, Y_train, X_test, Y_test = load_mnist(dataset_root_path)
        # X_test = [normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) for x in X_test]
        # X_test = normalize(X_test)
        test_set = [(X_test[i], Y_test[i]) for i in range(len(X_test))]
        choice_idx = np.random.choice(len(test_set), 1000, replace=False)
        test_set = [test_set[i] for i in choice_idx]
        correct_num = 0
        for instance in test_set:
            y_hat = self.forward(instance[0])
            if np.argmax(y_hat) == instance[1]:
                correct_num += 1
        accuracy = correct_num / len(test_set)
        print(f"Test accuracy: {accuracy}")
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
