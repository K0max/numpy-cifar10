from denseNet import *
from utils import *
import pickle
import os

model_dir_path = "./model"
cifar_dataset_root_path = "./cifar-10-batches-py"
mnist_dataset_root_path = "./mnist"

dataset = "cifar"
create_new_model = False
if dataset == "cifar":
    model_path = os.path.join(model_dir_path, "cifar_model.pkl")
    img_input_size = 3 * 32 * 32
    dataset_root_path = cifar_dataset_root_path
elif dataset == "mnist":
    model_path = os.path.join(model_dir_path, "mnist_model.pkl")
    img_input_size = 28 * 28
    dataset_root_path = mnist_dataset_root_path

model_path = "./model/denseNet-model.pkl"

def main():
    if os.path.exists(model_path) and not create_new_model:
        with open(model_path, 'rb') as f:
            net = pickle.load(f)
            print("Model loaded.")
    else:
        # fc_layer1 = fc_layer(x_size=img_input_size, y_size=512)
        # fc_layer2 = fc_layer(x_size=512, y_size=64)
        # # softmax is included in cross_entropy
        # fc_layer3 = fc_layer(x_size=64, y_size=10, activation=None)
        # net = denseNet(layers=[fc_layer1, fc_layer2, fc_layer3])
        fc_layer1 = fc_layer(x_size=img_input_size, y_size=512)
        fc_layer2 = fc_layer(x_size=512, y_size=10)
        net = denseNet(layers=[fc_layer1, fc_layer2])
        print("New model created.")
    # net.train(learning_rate=1e-3, batch_size=100, num_epoch=10, dataset_root_path=dataset_root_path)
    net.test(dataset_root_path)
    net.save_model(model_path)

if __name__ == "__main__":
    main()