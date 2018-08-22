import numpy as np
import tensorflow as tf
from config import args
from sklearn.model_selection import train_test_split

class CNN:
    def __init__(self,lr=0.001,embeddng_size=300,word2idx=None,lr_decay=0.99,decay_size=200,moving_average_decay=0.99,
                 filters_w=[3,4,5],num_filters=100,activation=tf.nn.relu,grad_clip=1.0,sess=None,seq_len=args.max_len,
                 num_class=None,path="./trained/"):
        self.lr=lr
        self.embeddng_size=embeddng_size
        self.word2idx=word2idx
        self.num_class=num_class
        self.lr_decay=lr_decay
        self.decay_size=decay_size
        self.moving_average_decay=moving_average_decay
        self.filters_w=filters_w
        self.num_filters=num_filters
        self.activation=activation
        self.grad_clip=grad_clip
        self.path=path
        self.sess=sess
        self.seq_len=seq_len
        with tf.variable_scope("TextCNN",reuse=False):
            self.add_input()
            self.add_embedding()
            self.add_conv()
            self.add_project()
            self.add_loss()
        with tf.variable_scope("TextCNN", reuse=True):
            self.add_pred()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables())

    def save_model(self,path):
        self.saver.save(self.sess, save_path=path)

    def load_model(self,path):
        self.saver.restore(sess=self.sess, save_path=path)
        print("model loaded....")

    def add_input(self):
        self.X=tf.placeholder(tf.int32,[None,self.seq_len])
        self.Y=tf.placeholder(tf.int32,[None])

    def add_embedding(self):
        embedding = tf.get_variable('embedding', [len(self.word2idx), self.embeddng_size], tf.float32,
                                        tf.random_uniform_initializer(-1.0, 1.0))

        embedded = tf.nn.embedding_lookup(embedding, self.X)
        self.input=tf.expand_dims(embedded,-1)
        #print("input:",self.input.shape)#(?, 200, 200, 1)
    def add_conv(self):
        pooled_outputs = []
        total_filters=self.num_filters*len(self.filters_w)
        #seq_len=self.input.get_shape().as_list()[1] #此处返回的是NoneType
        for i,kernel_w in enumerate(self.filters_w):
            kernel=[kernel_w,self.embeddng_size]
            #[B,seq-kw+1,1,num_filters]
            conv_out=tf.layers.conv2d(inputs=self.input,filters=self.num_filters,
                                      padding="valid",kernel_size=kernel,activation=self.activation)

            pool_out=tf.layers.max_pooling2d(inputs=conv_out,pool_size=[self.seq_len-kernel_w+1,1],strides=1)
            pooled_outputs.append(pool_out)
            concat=tf.concat(pooled_outputs,axis=3)
        #print("pooled:",self.out.shape)#(?, 1, 1, 300)
        self.out=tf.reshape(concat,[-1,total_filters])#[B,total_filters]
        #print("out:",self.out.shape)
    def add_project(self,is_train=True):
        #self.project=tf.layers.dense(self.out,50,activation=self.activation)
        self.drop=tf.layers.dropout(self.out,rate=0.5,training=is_train)
        self.final=tf.layers.dense(self.drop,self.num_class,activation=self.activation)#[b,K]
        #print("final",self.final)

    def add_pred(self,is_train=False):
        self.drop=tf.layers.dropout(self.out,rate=0.5,training=is_train)
        self.pred = tf.layers.dense(self.drop, self.num_class, activation=self.activation)

    def add_loss(self):
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.final,labels=self.Y))

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_exp = tf.train.exponential_decay(self.lr, self.global_step, self.decay_size,
                                                            self.lr_decay)
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
        # train_op=optmizer
        # grad_vars=optmizer.compute_gradients(self.loss,params)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip)
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_step)
        variables_averages_op = var_avg.apply(params)

        with tf.control_dependencies([gard_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')

    def train(self,X,Y,batch_size,seed):
        #X,Y=shuffle(X, Y, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
        for epoch in range(100):
            print("starting epoch: ",epoch)
            for step,(X_batch,Y_batch) in enumerate(self.gen_batch(X_train,y_train,batch_size=batch_size)):
                logits,loss,_,lr,gl=self.sess.run([self.final,self.loss,self.train_op,self.learning_rate_exp,self.global_step],feed_dict={self.X:X_batch,self.Y:Y_batch})
                print("Epoch %d/%d | |Batch %d/%d | train_loss: %.8f | gl_step : %d|lr: %.6f"
                      % (epoch+1, 100, step + 1, int(len(X) // batch_size)+1, loss,gl,lr))
            #with tf.variable_scope("TextCNN", reuse=True):
            acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.final,axis=1),y_test),tf.float32))
            test_stack,acc=self.sess.run([self.pred,acc],feed_dict={self.X:X_test,self.Y:y_test})
            train_stack=self.sess.run(self.final,feed_dict={self.X:X_train,self.Y:y_train})
            print("Accuracy for epoch %d : %.8f " %(epoch,acc))
        self.save_model(self.path)
        print("model saved......")
        return np.argmax(train_stack,axis=1).reshape((-1,1)),np.argmax(test_stack,axis=1).reshape((-1,1))

    def predict(self,X):
        input=[self.word2idx.get(x,self.word2idx['<unk>']) for x in X.split()]
        if len(input)<self.seq_len:
            input=input+[self.word2idx['<pad>']]*(self.seq_len-len(input))
        else:
            input=input[:self.seq_len]
        input=np.array(input).reshape((-1,self.seq_len))
        pred = self.sess.run(self.pred, feed_dict={self.X: input})
        return np.array(np.argmax(pred,axis=1))

    def gen_batch(self,X,Y,batch_size):
        for i in range(0,int(len(X)/batch_size)+1):
            yield X[i*batch_size:(i+1)*batch_size],Y[i*batch_size:(i+1)*batch_size]

