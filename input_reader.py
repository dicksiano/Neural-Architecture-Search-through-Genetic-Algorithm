import numpy as np

NUM_INPUTS = 3
NUM_OUTPUTS = 1
TRAIN_PERCENTAGE = 0.9

def read_data():
    lines = open('dataKickDecision_Final.txt', 'r').readlines()
    np.random.shuffle(lines) # Shuffling data
    x, y = [], []

    for line in lines:
        vals = line.split()
        x.append( [ float(x) for x in vals[:-1] ] )
        y.append( float( vals[-1]  ) )

    x = np.array(x)
    y = np.array(y)

    assert( x.shape[0] == y.shape[0] )
    assert( x.shape[1] == NUM_INPUTS )

    train_idx = int(len(x) * TRAIN_PERCENTAGE)  # Split data between train/test
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test,  y_test  = x[train_idx:], y[train_idx:]

    return [x_train, y_train, x_test, y_test]