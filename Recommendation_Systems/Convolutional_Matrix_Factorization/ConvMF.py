import numpy as np
import tensorflow as tf

class CNN:
    def __init__(self,lr=0.01,embeddng_size=200,vocab_size=8001,lr_decay=0.9,decay_size=1000,moving_average_decay=0.99,
                 K=[200,50],filters_w=[3,4,5],num_filters=100,activation=tf.nn.relu,grad_clip=5.0,sess=None,seq_len=None):
        self.lr=lr
        self.embeddng_size=embeddng_size
        self.vocab_size=vocab_size
        self.lr_decay=lr_decay
        self.decay_size=decay_size
        self.moving_average_decay=moving_average_decay
        self.K=K
        self.filters_w=filters_w
        self.num_filters=num_filters
        self.activation=activation
        self.grad_clip=grad_clip
        self.sess=sess
        self.seq_len=seq_len
        with tf.variable_scope("ConvMF",reuse=False):
            self.add_input()
            self.add_embedding()
            self.add_conv()
            self.add_project()
            self.add_loss()
        self.sess.run(tf.global_variables_initializer())

    def add_input(self):
        self.X=tf.placeholder(tf.int32,[None,self.seq_len])
        self.V=tf.placeholder(tf.float32,[None,self.K[1]])

    def add_embedding(self):
        embedding = tf.get_variable('embedding', [self.vocab_size, self.embeddng_size], tf.float32,
                                    tf.random_uniform_initializer(-1.0, 1.0))
        embedded = tf.nn.embedding_lookup(embedding, self.X)
        self.input=tf.expand_dims(embedded,-1)
        #print("input:",self.input.shape)#(?, 200, 200, 1)
    def add_conv(self):
        pooled_outputs = []
        total_filters=self.num_filters*len(self.filters_w)
        #seq_len=self.input.get_shape().as_list()[1]
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
        print("out:",self.out.shape)
    def add_project(self):
        self.project=tf.layers.dense(self.out,self.K[0],activation=self.activation)
        self.drop=tf.layers.dropout(self.project,rate=0.2)
        self.final=tf.layers.dense(self.drop,self.K[1],activation=self.activation)#[b,K]

    def get_project(self,X,reuse):
        with tf.variable_scope("ConvMF",reuse=reuse):
            theta=self.sess.run(self.final,feed_dict={self.X:X})
        return theta

    def add_loss(self):
        self.loss=tf.reduce_mean(tf.reduce_sum(tf.pow(self.V-self.final,2)))
        #self.loss=tf.reduce_sum(tf.abs(self.V-self.final))
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

    def train(self,epoch,X,V,batch_size,seed):
        np.random.seed(seed)
        X = np.random.permutation(X)
        np.random.seed(seed)
        V = np.random.permutation(V)
        total_loss=0
        print("starting epoch: ",epoch)
        for step,(X_batch,V_batch) in enumerate(self.gen_batch(X,V,batch_size=batch_size)):
            loss,_,final=self.sess.run([self.loss,self.train_op,self.final],feed_dict={self.X:X_batch,self.V:V_batch})
            print("Epoch %d/%d | |Batch %d/%d | train_loss: %.10f"
                  % (epoch+1, 10, step + 1, int(len(X) // batch_size)+1, loss))
            total_loss+=loss
        final_loss = total_loss
        print("ending epoch:%d | project_loss(total_loss on entire batch): %.3f "%(epoch+1,final_loss))
        return final_loss


    def gen_batch(self,X,V,batch_size):
        for i in range(0,int(len(X)/batch_size)+1):
            yield X[i*batch_size:(i+1)*batch_size],V[i*batch_size:(i+1)*batch_size]




