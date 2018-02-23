from RBFN import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('ballist.dat', 'rb') as data:
        ball_data = np.loadtxt(data, delimiter=" ", skiprows=0)
        x_train = ball_data[:, 0:2]
        y_train = ball_data[:, 2:4]
        
    with open('balltest.dat', 'rb') as data:
        ball_test = np.loadtxt(data, delimiter=" ", skiprows=0)
        x_test = ball_data[:, 0:2]
        y_test = ball_data[:, 2:4]


    r = RBFN()
    # parameter settings for rbfn
    nodes = 25
    learning_rule = 'delta'
    batch = False
    epochs = 500
    eta = 0.01
    strategy = 'competitive'
    vec_sigma = [0.5] * nodes
    normalize = False
    sigma_start, sigma_end, sigma_step = 0.1, 1, 0.1
    nodes_start, nodes_end, nodes_step = 10, 14 , 1
    NODES = list(np.arange(nodes_start,nodes_end,1))
    SIGMAS = list(np.arange(sigma_start,sigma_end,sigma_step))
    err_mat_0 = np.zeros((len(NODES),len(SIGMAS)))
    err_mat_1 = np.zeros((len(NODES),len(SIGMAS)))
    for i,nodes in enumerate(NODES):
        for j,sigma in enumerate(SIGMAS):
            r = RBFN()
            vec_sigmas = [sigma] * nodes
            r.train(x_train, y_train, nodes, vec_sigma,
                    learning_rule, batch,
                    epochs, eta, strategy, normalize)
            y = r.predict(x_test)
            err = r.error(y_test,y)
            err_mat_0[i,j], err_mat_1[i,j] = err[0], err[1] 
            print(err_mat_0)
            print(err_mat_1)
    
    best_settings_0 = np.unravel_index(err_mat_0.argmin(),err_mat_0.shape)
    print(best_settings_0)
    vec_sigmas = [ SIGMAS[best_settings_0[1]] ] * NODES[best_settings_0[0]]
    r = RBFN()
    r.train(x_train, y_train, NODES[best_settings_0[0]], vec_sigmas,
            learning_rule, batch,
            epochs, eta, strategy, normalize)
    y = r.predict(x_test)

    plt.title( "Col0. nodes:"+str(NODES[best_settings_0[0]])+" sigma:" +str(SIGMAS[best_settings_0[1]]))
    plt.plot(y_test[:, 0], c="b", label="test_data")
    plt.plot(y[:, 0], c="r", label="model")
    plt.legend()
    plt.show()


    r = RBFN()
    best_settings_1 = np.unravel_index(err_mat_1.argmin(),err_mat_1.shape)
    print(best_settings_1)
    vec_sigmas = [ SIGMAS[best_settings_1[1]] ] * NODES[best_settings_1[0]]
    
    r.train(x_train, y_train, NODES[best_settings_1[0]], vec_sigmas,
            learning_rule, batch,
            epochs, eta, strategy, normalize)
    y = r.predict(x_test)
    
    plt.title( "Col1. nodes:"+str(NODES[best_settings_1[0]])+" sigma:" +str(SIGMAS[best_settings_1[1]]))
    plt.plot(y_test[:, 1], c="b", label="test_data")
    plt.plot(y[:, 1], c="r", label="model")
    plt.legend()
    plt.show()












#     r.train(x_train, y_train,28, [0.1]*23,
#             learning_rule, batch,
#             epochs, eta, strategy, normalize)
#     y = r.predict(x_test)
#     print(r.error(y,y_test))
#     plt.title( "Col0. nodes:"+str(28)+" sigma:" +str(0.1))
#     plt.plot(y_test[:, 0], c="b", label="test_data")
#     plt.plot(y[:, 0], c="r", label="model")
#     plt.legend()
#     plt.show()
# 
# 
# 
#     r.train(x_train, y_train,23, [0.7]*23,
#             learning_rule, batch,
#             epochs, eta, strategy, normalize)
#     y = r.predict(x_test)
#     print(r.error(y,y_test))
#     plt.title( "Col1. nodes:"+str(23)+" sigma:" +str(0.7))
#     plt.plot(y_test[:, 1], c="b", label="test_data")
#     plt.plot(y[:, 1], c="r", label="model")
#     plt.legend()
#     plt.show()
# 
# 



if __name__ == '__main__':
    main()
