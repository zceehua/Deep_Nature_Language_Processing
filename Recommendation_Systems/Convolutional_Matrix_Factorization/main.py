from config import args
from data import Data
import numpy as np
import tensorflow as tf
from ConvMF import CNN
from update import update_UV

if __name__ == '__main__':
    np.random.seed(1234)
    context_file = "mult.dat"
    train_R = "cf-train-1-users.dat"
    test_R = "cf-test-1-users.dat"

    lambda_u = args.lambda_u  # lambda_u in CDL
    lambda_v = args.lambda_v  # lambda_v in CDL
    K = args.K
    num_iter = args.num_epochs
    batch_size = args.batch_size
    max_len = args.max_len
    epochs=args.epochs
    embedding=args.embedding
    lr=args.lr
    if args.activation=="relu":
        activation=tf.nn.relu

    data_loader = Data(context_file=context_file)
    R = data_loader.read_user(train_R)
    X = data_loader.read_context()
    print(len(X))
    X = data_loader.pad_input(X, max_len, int(8000))

    g=tf.get_default_graph()
    sess=tf.Session(graph=g)

    model = CNN(lr=0.01, embeddng_size=200, activation=activation, sess=sess,seq_len=max_len)
    V=model.get_project(X,reuse=False)
    theta=V
    #print(V[:128,:])
    U = np.random.uniform(size=(R.shape[0], K))#(5551, 16980)
    #print(U.shape)

    for i in range(epochs):
         U,V,E_loss=update_UV(R,U,V,theta,lambda_u, lambda_v,K)
         proj_loss = model.train(i, X, V, batch_size, seed=123)
         theta = model.get_project(X, reuse=True)
         loss=E_loss+lambda_v*proj_loss/2
         print("E_loss:%.5f | proj_loss:%.5f | total_loss: %.5f " % (E_loss, proj_loss,loss))






