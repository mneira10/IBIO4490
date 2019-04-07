
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import tqdm 
import os 

def sigmoid(x):
    return 1/(1+np.exp(-x))


def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ", num_of_instances)
    print("instance length: ", len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1, num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion) == 3 else 0  # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    # data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test


class Model():
    def __init__(self):
        params = 48*48  # image reshape
        out = 1  # smile label
        self.lr = 0.001  # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))
                                                    ) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr


def train(model, x_train, y_train, x_test, y_test,lr=0.001,bs = 100):

    print('training model with lr={} and bs={}'.format(lr,bs))

    batch_size = bs  # Change if you want
    model.lr = lr
    epochs = 20000  # Change if you want
    tot_train_loss = []
    tot_test_loss = []
    for i in tqdm.tqdm(range(epochs),total=epochs):
        loss = []
        for j in range(0, x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_test)
        loss_test = model.compute_loss(out, y_test)
        loss_train = np.array(loss).mean()
        # print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(
        #     i, loss_train, loss_test))
        tot_train_loss.append(loss_train)
        tot_test_loss.append(loss_test)
    plot(tot_train_loss, tot_test_loss, model.lr)


def plot(train_loss, test_loss, lr):  # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses
    assert len(train_loss) == len(test_loss)
    epochs = [x+1 for x in range(len(train_loss))]
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning rate = {}'.format(lr))
    plt.legend()
    
    plt.savefig('./lr_{}_bs_{}/train_test_loss.png'.format(lr,bs))
    plt.close()
    


def test(model, x_train, y_train, x_test, y_test,lr,bs):
    # _, _, x_test, y_test = get_data()
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
    tot_test_loss = []

    
    out = model.forward(x_test)

    threshold = np.arange(0,1,0.01)
    precisions = []
    recalls = []
    for t in threshold:
        CM = np.zeros((2, 2))
        pred = np.abs(out-1) < t
        # print(t)
        for i in range(len(out)):
            if pred[i][0] == y_test[i][0] and y_test[i][0] == 1:
                CM[0][0] += 1
            elif pred[i][0] == y_test[i][0] and y_test[i][0] == 0:
                CM[1][1] += 1
            elif pred[i][0] != y_test[i][0] and y_test[i][0] == 1:
                CM[1][0] += 1
            else:
                CM[0][1] += 1
        # print(CM)
        precision = CM[0][0]/CM[0][1]
        recall = CM[0][0]/CM[1][0]
        precisions.append(precision)
        recalls.append(recall)

    # print(precisions)
    # print(recalls)
    plt.plot(recalls,precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('./lr_{}_bs_{}/pr_curve.png'.format(lr,bs))

    plt.close()



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()

    lrs = [0.001,0.0001,0.00001,0.000001]
    for learnRate in lrs:
        model = Model()
        lr = learnRate
        bs = 100
        os.system('mkdir lr_{}_bs_{}'.format(lr,bs))
        train(model, x_train, y_train, x_test, y_test,lr,bs)
        test(model, x_train, y_train, x_test, y_test,lr,bs)
