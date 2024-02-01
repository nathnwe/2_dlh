# Import dependencies
# Here we load the packages that we need for the rest of the practical
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

###########################################################################################

def centring(X):
    print(X.shape)
    epsilon = 1e-7 # To prevent division by 0
    X = (X -X.mean(axis=0,keepdims=True)) / (X.std(axis=0,keepdims=True)+epsilon)
    return X

def to_one_hot(y, num_classes):
    y = y.squeeze()
    store = np.zeros((y.shape[0], num_classes))
    for c in range(0, num_classes):
         store[:,c][y==c] = 1 
    return store

def prepare_data(pth_to_data):
    print('Loading Data')
    # First we load the data
    dataset = np.load(pth_to_data)
    X_train, y_train = dataset['train_images'], dataset['train_labels']
    X_val, y_val = dataset['val_images'], dataset['val_labels']
    X_test, y_test = dataset['test_images'], dataset['test_labels']

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    print('Data Splits')
    print('Train: \t Label 0 = ', np.sum(np.where(y_train == 0, 1, 0)), '\t Label 1 = ', np.sum(np.where(y_train == 1, 1, 0)))
    print('Val: \t Label 0 = ', np.sum(np.where(y_val == 0, 1, 0)), '\t Label 1 = ', np.sum(np.where(y_val == 1, 1, 0)))
    print('Test: \t Label 0 = ', np.sum(np.where(y_test == 0, 1, 0)), '\t Label 1 = ', np.sum(np.where(y_test == 1, 1, 0)))
    
    print('Centring Data')
    X_train = centring(X_train)
    X_val = centring(X_val)
    X_test = centring(X_test)
    
    X_train = np.reshape(X_train, (-1, 1, 28, 28))
    X_val = np.reshape(X_val, (-1, 1, 28, 28))
    X_test = np.reshape(X_test, (-1, 1, 28, 28))

    print('Converting labels to one hot')
    y_train = to_one_hot(y_train, 2)
    y_val = to_one_hot(y_val, 2)
    y_test = to_one_hot(y_test, 2)

    print('X_data: ', X_train.shape, X_val.shape, X_test.shape)
    print('y data: ', y_train.shape, y_val.shape, y_test.shape)

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]
    