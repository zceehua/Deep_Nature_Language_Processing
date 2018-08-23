import tensorflow as tf
import numpy as np
from config import args

class baseModel(object):
    def __init__(self,embedding,num_class,embedding_size,vocab_size):
        self.num_class=num_class
        self.embedding_size=embedding_size
        self.embedding=embedding
        self.vocab_size=vocab_size
        self.build_graph()

    def build_graph(self):
        with tf.device("/cpu:0"):#my PC has bug with gpu:0, you can comment this line
            self.add_placeholder()
            sent1,sent2=self.add_embedding()
            alpha,beta=self.add_attend(sent1,sent2)
            v1=self.add_compare(sent1,beta,False)
            v2=self.add_compare(sent2,alpha,True)
            self.add_aggregate(v1,v2)
            self.add_backward()
            self.sess=tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())


    def add_placeholder(self):
        self.sent1=tf.placeholder(tf.int32,[None,args.max_len])
        self.sent2=tf.placeholder(tf.int32,[None,args.max_len])
        self.sent1_len=tf.placeholder(tf.int32,[None])
        self.sent2_len=tf.placeholder(tf.int32,[None])
        self.label=tf.placeholder(tf.int64,[None])
        self.is_training=tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)


    def add_embedding(self):
        if self.embedding.any():
            word_embedding=tf.Variable(initial_value=self.embedding,trainable=False)
        else:
            word_embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], tf.float32)
        sent1=tf.nn.embedding_lookup(word_embedding,self.sent1)
        sent2=tf.nn.embedding_lookup(word_embedding,self.sent2)
        sent1=self.input_transform(sent1,False)
        sent2=self.input_transform(sent2,True)#(B,T,D)
        return sent1,sent2

    def input_transform(self,sent,reuse):
        if args.proj:
            sent=self.embedding_proj(sent,reuse)
        else:
            sent=sent
        return sent

    def embedding_proj(self,embedding,reuse=False):
        T=embedding.get_shape()[1]
        #T=tf.shape(embedding)[1] not working
        with tf.variable_scope("embed_proj",reuse=reuse,initializer=tf.random_normal_initializer(0.0, 0.1),
                               regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
            project=tf.layers.dense(tf.reshape(embedding,[-1,self.embedding_size]),args.num_units)

        project=tf.reshape(project,[-1,T,args.num_units])
        return project

    def add_attend(self,sent1,sent2):#(B,T,D)
        with tf.variable_scope("inter_atten") as self.inter_scope:
            sent1=self.attend_transform(sent1,self.inter_scope,args.num_units,False)
            sent2=self.attend_transform(sent2,self.inter_scope,args.num_units,True)
            attention=tf.matmul(sent1,sent2,transpose_b=True)#(B,T1,T2)
            #print(sent1,sent2)
            sent1_masked=self.mask(attention,self.sent2_len,-np.inf,"attend")#(B,T1,T2)
            sent2_masked=self.mask(tf.transpose(attention,[0,2,1]),self.sent1_len,-np.inf,"attend")#(B,T2,T1)
            attention1=self.softmax_atten(sent1_masked)
            attention2=self.softmax_atten(sent2_masked)

            beta=tf.matmul(attention1,sent2)#(B,T1,D)
            alpha=tf.matmul(attention2,sent1)
        return alpha,beta


    def add_compare(self,sent,attention,reuse):
        with tf.variable_scope("compare",reuse=reuse) as self.comp_scope:
            inputs=tf.concat([sent,attention],axis=2)
            v=self.compare_transform(inputs,self.comp_scope,args.num_units,reuse)#(B,T,D)
        return v

    def add_aggregate(self,sent1,sent2):
        with tf.variable_scope("aggregate") as self.agg_scope:
            masked_1=self.mask(sent1,self.sent1_len,0,"agg")
            masked_2=self.mask(sent2,self.sent2_len,0,"agg")
            v1=tf.reduce_sum(masked_1,axis=1)
            v2=tf.reduce_sum(masked_2,axis=1)
            v=tf.concat([v1,v2],axis=-1)
            output=self.MLP(v,self.agg_scope,args.num_units)
        self.logits=tf.layers.dense(output,self.num_class,kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))

    def add_backward(self):
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label))
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss+=reg_losses
        self.learning_rate_exp = tf.train.exponential_decay(args.lr, self.global_step,
                                                            args.decay_size,
                                                            args.decay_factor)
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.grad_clip)
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(args.moving_average_decay, self.global_step)
        variables_averages_op = var_avg.apply(params)
        with tf.control_dependencies([gard_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')

    def mask(self,value,sent_size,mask_value,mode):
        mask=tf.ones_like(value)*mask_value
        #try to use get_shape instead of tf.shape
        T,D=value.get_shape()[1],value.get_shape()[2]
        if mode=="attend":
            mask_sent = tf.sequence_mask(sent_size, D)
            mask_sent=tf.tile(tf.expand_dims(mask_sent,1),[1,T,1])
        elif mode=="agg":
            mask_sent=tf.sequence_mask(sent_size,T)
            mask_sent = tf.tile(tf.expand_dims(mask_sent, -1), [1, 1, D])

        masked=tf.where(mask_sent,value,mask)
        return masked

    def softmax_atten(self,value):
        # D=tf.shape(value)[-1]
        # atten=tf.reshape(value,[-1,D])
        # atten=tf.reshape(tf.nn.softmax(atten),tf.shape(value))
        return tf.nn.softmax(value)

    def attend_transform(self,sent,scope,num_units,reuse):
         raise NotImplementedError()

    def compare_transform(self,sent,scope,num_units,reuse):
        raise NotImplementedError()

    def MLP(self,inputs,scope,num_units,reuse=False):
        with tf.variable_scope(scope,reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
            inputs=tf.layers.dropout(inputs,rate=args.dropout)
            inputs=tf.layers.dense(inputs,num_units,tf.nn.relu)
            inputs = tf.layers.dropout(inputs, rate=args.dropout)
            output = tf.layers.dense(inputs, num_units,tf.nn.relu)
        return output

    def train(self,train,val):
        total = len(train["sent1"]) // args.batch_size + 1
        for epoch in range(args.num_epochs):
            count=0
            for step, data in enumerate(self.gen_batch(train, args.batch_size)):
                feed_dict = {self.sent1: data["sent1"], self.sent2: data["sent2"],
                             self.label: data["labels"], self.sent1_len:data["sent1_len"],
                             self.sent2_len:data["sent2_len"],self.is_training: True}
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f"
                      % (epoch + 1, args.num_epochs, step + 1, total, loss))

                if step % 1000 == 0 and step != 0:
                    print("validating......")
                    total_correct=0
                    total_loss=0
                    count=0
                    total_val = len(val["sent1"]) // args.batch_size + 1
                    print("total val steps: ",total_val)
                    for data_val in self.gen_batch(val, args.batch_size):

                        feed_dict = {self.sent1: data_val["sent1"], self.sent2: data_val["sent2"],
                                     self.label: data_val["labels"], self.sent1_len: data_val["sent1_len"],
                                     self.sent2_len: data_val["sent2_len"], self.is_training: False}
                        correct=tf.reduce_sum(tf.cast(tf.equal(self.label,tf.argmax(self.logits,-1)),tf.float32))
                        val_loss,acc=self.sess.run([self.loss,correct],feed_dict=feed_dict)
                        total_correct+=acc
                        total_loss+=val_loss
                        count+=1
                        print(acc)

                    print("val_loss: %.3f | val_acc: %.3f" %(total_loss/count,total_correct/len(val["sent1"])))


    def gen_batch(self,X,batch_size):
        data={}
        for i in range(0,len(X["sent1"])//batch_size+1):
            data["sent1"]=X["sent1"][i*batch_size:(i+1)*batch_size]
            data["sent2"]=X["sent2"][i*batch_size:(i+1)*batch_size]
            data["labels"]=X["label"][i*batch_size:(i+1)*batch_size]
            data["sent1_len"]=X["sent1_len"][i*batch_size:(i+1)*batch_size]
            data["sent2_len"]=X["sent2_len"][i*batch_size:(i+1)*batch_size]
            yield  data