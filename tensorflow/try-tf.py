# Tensorflow
import tensorflow as tf
s = tf.InteractiveSession()

# scikit-learn
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cross_validation import train_test_split

# Numpy ans Matplotlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# change MPL style
#mpl.style.use('fivethirtyeight')

# Configuration variables:
num_labels = 2   # The number of labels
NUM_EPOCHS = 5
NUM_HIDDEN = 5
BATCH_SIZE = 100  # The number of training examples to use per training step
n_train = 1000
n_test = 200

def generate_dataset(func,
                     n_train=n_train,
                     n_test=n_test,
                     num_labels=num_labels,
                     **kwargs):
    """ Create synthetic classification data-sets

    Parameters
    ----------
    func : one of {'make_blobs', 'make_circles', 'make_moons'}
        What kind of data to make.
    n_train : int
        The size of the training set.
    num_labels : int
        The number of classes.

    Returns
    -------
    train_data, test_data : 2D arrays
        Dimensions: {n_train, n_test} by 2
    train_labels, test_labels: one-hot encoder arrays
        These have dimensions {n_train, n_test} by num_labels
    """
    fvecs, labels = func(n_train + n_test, **kwargs)
    # We need the one-hot encoder !
    labels_onehot = (np.arange(num_labels) == labels[:, None])
    train_data, test_data, train_labels, test_labels = train_test_split(fvecs.astype(np.float32),
                                                                        labels_onehot.astype(np.float32),
                                                                        train_size=n_train)
    return train_data, test_data, train_labels, test_labels

train_data, test_data, train_labels, test_labels = generate_dataset(make_blobs, n_train=1000, n_test=200, centers=2, center_box=[-4., 4.])

print train_labels.shape

fig, ax = plt.subplots(1)
ax.plot(train_data[np.where(train_labels[:, 0]), 0],
        train_data[np.where(train_labels[:, 0]), 1], 'bo')
ax.plot(train_data[np.where(train_labels[:, 1]), 0],
        train_data[np.where(train_labels[:, 1]), 1], 'ro')
ax.set_aspect('equal')

def train_softmax(train_data, train_labels,
                  batch_size=BATCH_SIZE,
                  num_epochs=NUM_EPOCHS,
                  sess=s) :
    """
    Train a softmax network with cross-entropy penalty

    Parameters
    ----------
    train_data, train_labels: output of 'generate_dataset'.
    batch_size : int
        The number of items in training batch.
    num_epochs : int
        The number of training epochs.

    Returns
    -------
    W : Variable
        The model estimated weight matrix.
    b : Variable
        The model estimated bias vector.
    s : TensorFlow session
        The session in which these variables were defined.
    """
    train_size, num_features = train_data.shape
    train_size, num_labels = train_labels.shape
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # traning step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None, num_labels])

    # Define and initialize the network:
    # These are theweights that inform how each feature contribute to
    # the classification
    W = tf.Variable(tf.zeros([num_features, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

    # Optimization:
    cross_entropy = -tf.reduce_sum(y * tf.log(y_hat))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    #s = tf.Session()
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run(session=sess)
    print('Initialized!')

    # Iterate and train:
    print('Training: ')
    for step in xrange(num_epochs * train_size // batch_size):
        print(step),

        offset = (step * batch_size) % train_size
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size)]
        train_step.run(feed_dict={x: batch_data, y: batch_labels}, session=sess)

        if offset >= train_size-batch_size:
            print
    return W, b

W, b = train_softmax(train_data, train_labels)

def evaluate_softmax(W, b, test_data, test_labels, sess=s):
    """
    Proportion correct classification in test data of
    the softmax model.

    Parameters
    ----------
    W, b, s : outputs of 'train_softmax'
    test_data, test_labels: outputs of 'generate_dataset'

    Returns
    -------
    p_correct : float
        The proportion correct classificatioin
    """
    test_size, num_features = test_data.shape
    test_size, num_labels = test_labels.shape

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.placeholder("float", shape=[None, num_labels])
    # Evaluation :
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    p_correct = accuracy.eval(feed_dict={x: test_data, y: test_labels}, session=sess)
    return p_correct

print evaluate_softmax(W, b, test_data, test_labels, s)

Wx_b = np.dot(test_data, s.run(W)) + s.run(b)
cls = tf.nn.softmax(Wx_b)
print cls
cls_np = np.round(s.run(cls))

fig, ax = plt.subplots(1)
ax.plot(test_data[np.where(cls_np[:, 0]), 0],
        test_data[np.where(cls_np[:, 0]), 1], 'bo')
ax.plot(test_data[np.where(cls_np[:, 1]), 0],
        test_data[np.where(cls_np[:, 1]), 1], 'ro')
ax.set_aspect('equal')
