import numpy as np

import pickle, gzip

f = gzip.open('Data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding="latin")
f.close()


train_data = train_set[0]
train_labels = train_set[1]

valid_data = valid_set[0]
valid_label = valid_set[1]

test_data = test_set[0]
test_labels = test_set[1]


def add_ones(data):
    a, _ = np.shape(data)
    b = np.ones((a, 1))
    return np.hstack((data, b))

def one_hot(label):
    y = np.eye(10)[label]
    return y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return z * (1 - z)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def cost_func(out, expected):
    return (out - expected)

def create_weights():
    hidden_layer = np.random.randn(785, 101) * 1/np.sqrt(785)
    out_layer = np.random.randn(101, 10) * 1/np.sqrt(101)
    return hidden_layer, out_layer


def train(epochs, train_data, train_labels, batch_size, eta):
    w_1, w_2 = create_weights()
    
    train_data = add_ones(train_data)
    n = len(train_data)
    for j in range(epochs):
        for i in range(0, n, batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_label = train_labels[i:i+batch_size]

            # Sigmoid 
            # out_1 = sigmoid(np.dot(batch_data, w_1))
            # out_2 = sigmoid(np.dot(out_1, w_2))
            
            # layer2_err = cost_func(out_2, one_hot(batch_label))
            # layer2_delta = layer2_err * sigmoid_prime(out_2)

            # layer1_err = np.dot(layer2_delta, w_2.T)
            # layer1_delta = layer1_err * sigmoid_prime(out_1)

            # layer1_adjustment = np.dot(batch_data.T, layer1_delta)
            # layer2_adjustment = np.dot(out_1.T, layer2_delta)
            
            # w_1 -= layer1_adjustment * eta
            # w_2 -= layer2_adjustment * eta

            # Softmax
            out_1 = sigmoid(np.dot(batch_data, w_1))
            out_2 = softmax(np.dot(out_1, w_2))
            
            layer2_err = cost_func(out_2, one_hot(batch_label))
            layer2_delta = np.dot(out_1.T, layer2_err)

            layer1_err = np.dot(layer2_err, w_2.T)
            layer1_delta = layer1_err * sigmoid_prime(out_1)

            layer1_adjustment = np.dot(batch_data.T, layer1_delta)
            layer2_adjustment = layer2_delta
            
            w_1 -= layer1_adjustment * eta
            w_2 -= layer2_adjustment * eta

        print(j)
    return w_1, w_2

def test_val(test_data, w_1, w_2):
    out_1 = sigmoid(np.dot(test_data, w_1))
    out_2 = sigmoid(np.dot(out_1, w_2))
    return np.argmax(out_2)

def test_all(test_data, test_labels, w_1, w_2):
    count = 0
    for data, label in zip(test_data, test_labels):
        if test_val(data, w_1, w_2) == label:
            count += 1
    return count

w_1, w_2 = train(10, train_data, train_labels, 40, 0.05)

print()

test_data = add_ones(test_data)
print(test_all(test_data, test_labels, w_1, w_2))