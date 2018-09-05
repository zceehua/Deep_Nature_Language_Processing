import tensorflow as tf
import numpy as np
from config import args
from ModifiedLSTM import ModifiedLSTM

class MOS(object):
    def __init__(self,vocab_size=None):
        self.vocab_size = vocab_size
        with tf.variable_scope("MOS"):
            self.add_input()
            self.add_embedding()
            self.add_forward()
            self.add_backward()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def add_input(self):
        self.seq_len=tf.placeholder(tf.int32)
        self.input=tf.placeholder(tf.int32,[None,None])
        self.lables=tf.placeholder(tf.int32,[None,None])
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)


    def add_embedding(self):
        with tf.variable_scope("Embedding"):
            self.embedding=tf.get_variable("embedding_matrix",[self.vocab_size, args.emsize], tf.float32,tf.random_uniform_initializer(-1.0, 1.0))

        def f1():
            #drop word
            mask=tf.reshape(tf.multinomial(tf.log([[args.dropoute,1-args.dropoute]]),self.vocab_size),[self.vocab_size,1])
            mask=tf.cast(tf.tile(mask,[1,args.emsize]),tf.float32)#need to convert to float if we want consistent of data type
            self.embedding=self.embedding*mask
            return self.embedding
        def f2():
            return self.embedding

        #bugs with gpu version when is_training==False,any ideas?
        self.embedding=tf.cond(self.is_training,f1,f2)
        #drop embedding
        self.embedding=tf.nn.dropout(self.embedding,keep_prob=args.dropouti)
        self.input_embedding=tf.nn.embedding_lookup(self.embedding,self.input)#(B,T,300)

    def lstm_cell(self,size):
        return ModifiedLSTM(size,initializer=tf.orthogonal_initializer())

    def add_forward(self):
        cells=[self.lstm_cell(args.nhid if i!=args.nlayers-1 else args.nhidlast) for i in range(args.nlayers)]
        #print(cells)
        cells=[tf.nn.rnn_cell.DropoutWrapper(cells[i],output_keep_prob=args.dropouth ) if i!=args.nlayers-1 else cells[i] for i in range(args.nlayers)  ]
        #print(cells)
        cells=tf.nn.rnn_cell.MultiRNNCell(cells)
        self.initial_state=cells.zero_state(args.batch_size, tf.float32)

        self.raw_output,_=tf.nn.dynamic_rnn(cells,self.input_embedding,self.seq_len,self.initial_state)

        self.output=tf.nn.dropout(self.raw_output,keep_prob=args.dropout)#(B,T,650)
        latent=tf.nn.dropout(tf.layers.dense(self.output,args.n_experts*args.emsize,activation=tf.nn.tanh),keep_prob=args.dropoutl)##(B,T,10*300)
        if args.tied:
            with tf.variable_scope("Embedding",reuse=True):
                embedding=tf.transpose(tf.get_variable("embedding_matrix"),[1,0])
            #self.logits=tf.tensordot(tf.reshape(latent,[-1,args.emsize]),embedding,axes=[[-1],[0]])
            b = tf.get_variable('bias', [self.vocab_size], tf.float32, tf.constant_initializer(0.01))
            self.logits=tf.nn.xw_plus_b(tf.reshape(latent,[-1,args.emsize]),embedding,b)
        else:
            self.logits=tf.layers.dense(tf.reshape(latent,[-1,args.emsize]),self.vocab_size)#(B*T*10,300)==>(B*T*10,33278)


        self.pai=tf.reshape(tf.layers.dense(self.output,args.n_experts),[-1,args.n_experts])#(B*T,10)
        self.pai=tf.nn.softmax(self.pai)
        self.logits=tf.nn.softmax(tf.reshape(self.logits,[-1,self.vocab_size]))
        self.logits=tf.reshape(self.logits,[-1,args.n_experts,self.vocab_size])#(B*T,10,33278)

        prob=tf.reduce_sum(self.logits*tf.tile(tf.expand_dims(self.pai,2),[1,1,self.vocab_size]),axis=1)#(B*T,33278)
        self.prob=tf.reshape(prob,[-1,self.seq_len,self.vocab_size])

    def add_backward(self):
        self.raw_loss=tf.contrib.seq2seq.sequence_loss(self.prob,self.lables,tf.ones([args.batch_size,self.seq_len]))
        self.loss=self.raw_loss
        # Activiation Regularization
        self.loss+=tf.reduce_sum(args.alpha *tf.reduce_mean(tf.square(self.output),2))
        # Temporal Activation Regularization (slowness)
        self.loss+=tf.reduce_sum(args.beta *tf.reduce_mean(tf.square(self.raw_output[:-1]-self.raw_output[1:]),2))
        # self.learning_rate_exp = tf.train.exponential_decay(args.lr, self.global_step,
        #                                                     args.decay_size,
        #                                                     args.decay_factor)
        lr=args.lr*self.seq_len/args.bptt
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(lr)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.grad_clip)
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(args.moving_average_decay, self.global_step)
        variables_averages_op = var_avg.apply(params)
        with tf.control_dependencies([gard_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')


    def train(self,train,val):
        for epoch in range(args.num_epochs):
            for step,(inputs,targets,seq_len) in enumerate(self.gen_data(train)):
                feed_dict={self.seq_len:seq_len,self.input:inputs,self.lables:targets,self.is_training:True}
                loss,_=self.sess.run([self.raw_loss,self.train_op],feed_dict=feed_dict)
                print("Epoch %d/%d | Batch %d | train_loss: %.4f"
                      % (epoch + 1, args.num_epochs, step + 1, loss))

                if step % 10 == 0 and step != 0:
                    print("validating.....")
                    count=0
                    total_loss=0
                    for val_x,val_y,val_seq_len in self.gen_data(val):
                        feed_dict={self.seq_len:val_seq_len,self.input:val_x,self.lables:val_y,self.is_training:False}
                        loss= self.sess.run(self.raw_loss, feed_dict=feed_dict)
                        print("val_loss for Batch %d : %.4f"
                              % (count + 1,  loss))
                        count+=1
                        total_loss+=loss
                    print("averaged perplexity on val set: %.4f" % (total_loss/count))

    #This is exactly the same as validation process
    # def predict(self):
    #     pass

    def gen_data(self,X):
        # bptt is seq len:70
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))  # 每次抽取动态的seq len
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        max_len=X.shape[1]
        for i in range(max_len):
            seq_len=min(seq_len,max_len-1-i)
            yield X[:,i:i+seq_len],X[:,i+1:i+1+seq_len],seq_len


