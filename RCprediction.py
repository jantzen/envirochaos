# file: RCprediction.py

import numpy as np
from simpleRC import *
import matplotlib.pyplot as plt
import pdb


def prep_training_and_test(data, num_samples):

    print("Data shape: {}".format(data.shape))
    U = []
    y = []

    rows = data.shape[0] - num_samples
    for ii in range(rows):
        tmp_U = data[ii : ii + num_samples, :]
        tmp_y = data[ii + 1 : ii + num_samples + 1, :]
        U.append(tmp_U.flatten().reshape(1, -1))
        y.append(tmp_y.flatten().reshape(1, -1))
    U = np.concatenate(U, axis=0)
    y = np.concatenate(y, axis=0)
    print(U.shape, y.shape)

    # split into training (95%) and testing (5%)
    split = int(0.945 * U.shape[0])
    U_train = U[:split, :]
    y_train = y[:split, :]
    U_test = U[split:, :]
    y_test = y[split:, :]

    return U_train, y_train, U_test, y_test


def scale(X):
    # scales the input array X (assumed to have samples in rows) to have 0
    # mean and stdev=1
    mu = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X_scaled = (X - mu) / stdev
    return X_scaled, mu, stdev


def unscale(X, mu, stdev):
    # applies inverse transformation of scale
    return X * stdev + mu


def main(filename1='./data/2166184.csv', filename2='./data/2370691.csv',
        plot=True):
    num_samples = 500
    nn = 500
    sparsity = 0.01
    gamma = 0.1

    # open file for saving output
    f = open('prediction', 'w')
    print("Opening data file...")
    data = np.loadtxt('/mnt/data/envirochaos/LED_lightbulb_1/lb3_truncated.txt')
    data = data.reshape(-1, 1)
#    print("Normalizing data...")
#    data, mu, stdev = scale(data)

    if plot:
        # plot scaled data for verification
        plt.figure()
        plt.title('Full timeseries')
        t = np.arange(data.shape[0]) * 0.05 / 60.
        plt.plot(t, data)
        plt.xlabel('Time [min]')
        plt.ylabel('Inverse intensity [V]')
        plt.savefig('./lb3_fig1.jpg', dpi=600)


    print("Building RC...")
    rc = simpleRC(num_samples, nn, num_samples, sparsity=sparsity,
            mode='recurrent_forced')
    print("Revervoir size: {}".format(rc.Wres.shape))
#    f.write(("Revervoir size: {}\n".format(rc.Wres.shape)))

    print("Constructing training and testing datasets ...")
#    f.write("Constructing training and testing datasets for VA...\n")
    U_train, y_train, U_test, y_test = prep_training_and_test(data,
            num_samples)
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)
    t = t[num_samples:]
    t_train = t[:y_train.shape[0]]
    t_test = t[y_train.shape[0]:]
    if plot:
        plt.figure()
        plt.plot(t_train, y_train[:,0], t_test, y_test[:,0])
        plt.title('Division into training (blue) and test (orange)')
        plt.xlabel('Time [min]')
        plt.ylabel('Inverse intensity [V]')
        plt.savefig('./lb3_fig2.jpg', dpi=600)

    # test untrained accuracy
#    np.savetxt('steps', np.array(steps).reshape(1,1))
    steps = y_test.shape[0]
    U_init = U_train[0].reshape(-1,1)
#    preds = rc.run(U_init, steps)
#    error = np.sqrt(np.mean(np.linalg.norm((y_train[:steps,:] - preds), axis=1)))
#    print("Sanity check: untrained prediction accuracy = {}".format(error))
#    f.write("Sanity check: untrained prediction accuracy = {}\n".format(error))

    print("Training the RC ...")
#    f.write("Training the RC for VA...\n")
    rc.train(U_train, y_train, gamma=gamma)
    print("Testing the trained RC ...")
#    f.write("Testing the trained RC for VA...\n")
    preds = rc.run(U_init, steps)
    error = np.sqrt(np.mean(np.linalg.norm((y_train[:steps,:] - preds), axis=1)))
    print("Error on training set: {}".format(error))
#    f.write("Error on training set: {}\n".format(error))
    U_init = U_test[0,:].reshape(-1,1)
    preds = rc.run(U_init, steps)
    error = np.sqrt(np.mean(np.linalg.norm((y_test[:steps,:] - preds), axis=1)))
    print("Error on test set: {}".format(error))
#    f.write("Error on test set: {}\n".format(error))
#    print("Saving the linear output layer for VA...")
#    np.savetxt('Wout_VA', rc.Wout)
#    print("Saving the reservoir weights for VA...")
#    np.savetxt('W_VA', rc.Wres)
#    # unscale the data
#    y_units = unscale(y_test[:,-1], mu, stdev)
#    preds_units = unscale(preds[:,-1], mu, stdev)
#    np.savetxt('va_t', t)
#    np.savetxt('va_test', y_units)
#    np.savetxt('va_preds', preds_units)
    plt.figure()
    plt.title('Predictions on test data')
    plt.plot(t_test, y_test[:,0], 'k-', t_test[:steps], preds[:,0],
            'r--')
    plt.legend(['actual', 'predicted'])
    plt.xlabel('Time [min]')
    plt.ylabel('Inverse intensity [V]')
    plt.savefig('./lb3_fig3.jpg', dpi=600)

    if plot:
        plt.show()

if __name__ == '__main__':
    main()
