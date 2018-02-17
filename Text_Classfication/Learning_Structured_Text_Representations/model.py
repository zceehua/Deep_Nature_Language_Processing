import tensorflow as tf
import numpy as np
from config import args

class StructureModel(object):
    def __init__(self,embedding=None,vocab_size=None):
        self.vocab_size = vocab_size
        self.embedding = embedding
        with tf.variable_scope("Structure_Tree"):
            self.add_forward()
            self.add_backward()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def add_forward(self):
        self.build_input()
        self.add_embedding()
        self.add_weight()
        sent_atten_embed=self.add_sent_atten()
        self.add_doc_atten(sent_atten_embed)
        self.add_project()

    def build_input(self):
        self.input=tf.placeholder(tf.int32, [None, None, None])
        self.labels=tf.placeholder(tf.int64, [None])
        self.doc_len=tf.placeholder(tf.int32, [None])
        self.max_doc_len=tf.reduce_max(self.doc_len)
        #self.max_doc_len=60
        self.sent_len=tf.count_nonzero(self.input,axis=2,dtype=tf.int32)
        self.max_sent_len=tf.reduce_max(tf.placeholder(tf.int32,[None]))
        #self.max_sent_len=150
        self.is_training=tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)

    def add_embedding(self):
        with tf.variable_scope("Structure_Tree/embedding"):
            if self.embedding is None:
                word_embedding = tf.get_variable("embedding_matrix_w",[self.vocab_size, args.word_emb_size], tf.float32)
            else:
                word_embedding = tf.Variable(initial_value=self.embedding,name="embedding_matrix_w",dtype="float32")

            self.root_emb_sent=tf.get_variable("root_emb_sent",[1,1, 2*args.emb_sem], tf.float32)
            self.root_emb_doc=tf.get_variable("root_emb_doc",[1,1, 2*args.emb_sem], tf.float32)

            self.input_embedding=tf.nn.embedding_lookup(word_embedding,self.input)#(B,D,S,E)

    def add_weight(self):
        with tf.variable_scope("sent"):
            tf.get_variable("w_pc", [2 * args.emb_str, 2 * args.emb_str], dtype=tf.float32,)
            tf.get_variable("w_r", [2 * args.emb_str, 1], dtype=tf.float32)

        with tf.variable_scope("doc"):
            tf.get_variable("w_pc", [2 * args.emb_str, 2 * args.emb_str], dtype=tf.float32,)
            tf.get_variable("w_r", [2 * args.emb_str, 1], dtype=tf.float32)

    def rnn_cell(self):
        if args.rnn_type=="lstm":
            return tf.nn.rnn_cell.LSTMCell(args.emb_str + args.emb_sem)
        elif args.rnn_type=="gru":
            return tf.nn.rnn_cell.GRUCell(args.emb_str + args.emb_sem, kernel_initializer=tf.contrib.layers.xavier_initializer())

    def add_sent_atten(self):

        self.input_embedding=tf.reshape(self.input_embedding,[args.batch_size*self.max_doc_len,self.max_sent_len,args.word_emb_size])

        output, _ = tf.nn.bidirectional_dynamic_rnn(#ok
            cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn_cell() for _ in range(args.n_layers)]),
            cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn_cell() for _ in range(args.n_layers)]),
            inputs=self.input_embedding,
            sequence_length=tf.reshape(self.sent_len, [-1]),
            dtype=tf.float32,
            scope='bidirectional_rnn_sent')
        #each row is parent node word embedding
        self.token_sem=tf.concat([output[0][:,:,:args.emb_sem],output[1][:,:,:args.emb_sem]],2)#(B*T,S,2*75)
        self.token_str=tf.concat([output[0][:,:,args.emb_sem:],output[1][:,:,args.emb_sem:]],2)#(300, 100, 50)

        temp1 = tf.zeros([args.batch_size * self.max_doc_len, self.max_sent_len, 1], tf.float32)
        temp2 = tf.zeros([args.batch_size * self.max_doc_len, 1, self.max_sent_len], tf.float32)

        mask1 = tf.ones([args.batch_size * self.max_doc_len, self.max_sent_len, self.max_sent_len - 1], tf.float32)  # for sent level mask
        mask2 = tf.ones([args.batch_size * self.max_doc_len, self.max_sent_len - 1, self.max_sent_len], tf.float32)
        mask1 = tf.concat([temp1, mask1], 2)  # [:, :, 0] = 0，[batch_l * max_doc_l, max_sent_l,max_sent_l]
        mask2 = tf.concat([temp2, mask2], 1)  # [:, 0, :] = 0

        self.mask=mask2#ok
        self.w=self.input_embedding
        proot,pz=self.atten_score("sent",mask1,mask2)##(B*T,S,),(B*T,S,S)
        self.proot=proot#not ok
        self.root_emb_sent=tf.tile(self.root_emb_sent,[args.batch_size*self.max_doc_len,1,1])
        mask_sent = tf.reshape(tf.sequence_mask(self.sent_len, self.max_sent_len), [-1, self.max_sent_len])
        mask_sent = tf.to_float(tf.expand_dims(mask_sent, [-1]))#(B*T,S,1)

        sent_r=self.get_atten_vec("sent",self.token_sem,proot,pz,self.root_emb_sent,mask_sent)
        #pay attention to that we can not reshape as [args.batch_size, self.max_doc_len, -1], error may occur
        sent_r = tf.reshape(sent_r, [-1, self.max_doc_len, 3*args.emb_sem])  # [batch_l ,max_doc_l,75*3]
        self.sent_r=sent_r#not ok
        return sent_r

    #we could merge repeating code
    def add_doc_atten(self,sent_embed):
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn_cell() for _ in range(args.n_layers)]),
            cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                [self.rnn_cell() for _ in range(args.n_layers)]),
            inputs=sent_embed,
            sequence_length=self.doc_len,
            dtype=tf.float32,
            scope='bidirectional_rnn_doc')

        self.doc_sem = tf.concat([output[0][:, :, :args.emb_sem], output[1][:, :, :args.emb_sem]], 2)  # (B,T,2*75)
        self.doc_str = tf.concat([output[0][:, :, args.emb_sem:], output[1][:, :, args.emb_sem:]], 2)

        temp1 = tf.zeros([args.batch_size , self.max_doc_len, 1], tf.float32)
        temp2 = tf.zeros([args.batch_size , 1, self.max_doc_len], tf.float32)

        mask1 = tf.ones([args.batch_size , self.max_doc_len, self.max_doc_len - 1],tf.float32)  # for sent level mask
        mask2 = tf.ones([args.batch_size , self.max_doc_len - 1, self.max_doc_len], tf.float32)
        mask1 = tf.concat([temp1, mask1], 2)  # [:, :, 0] = 0，[batch_l , max_doc_l,max_doc_l]
        mask2 = tf.concat([temp2, mask2], 1)  # [:, 0, :] = 0
        proot, pz = self.atten_score("doc", mask1, mask2)

        self.root_emb_doc = tf.tile(self.root_emb_doc, [args.batch_size, 1, 1])
        mask_doc = tf.sequence_mask(self.doc_len, self.max_doc_len)
        mask_doc = tf.to_float(tf.expand_dims(mask_doc, [-1]))  # (B,T,1)

        doc_r = self.get_atten_vec("doc", self.doc_sem, proot, pz, self.root_emb_doc, mask_doc)
        self.doc_r = tf.reshape(doc_r, [args.batch_size,  3*args.emb_sem])  # [batch_l,75*3]


    def get_atten_vec(self,name,sem_vec,proot,pz,root_emb,mask):
        p = tf.matmul(pz, sem_vec, transpose_a=True) + tf.matmul(tf.expand_dims(proot, 2),
                                                                   root_emb)  # (B*T,S,2*75)
        c = tf.matmul(pz, sem_vec)  # (B*T,S,2*75)
        r_tmp = tf.reshape(tf.concat([p, c, sem_vec], 2), [-1, 6 * args.emb_sem])
        r = tf.layers.dense(r_tmp, 3 * args.emb_sem, activation=tf.nn.tanh)  # (B*T,S,3*75)
        if name=="sent":
            r=tf.reshape(r,[-1,self.max_sent_len,3*args.emb_sem])
        else:
            r = tf.reshape(r, [-1, self.max_doc_len, 3 * args.emb_sem])

        #  pooling
        if (args.sent_attention == 'sum'):
            r = r * mask
            r = tf.reduce_sum(r, 1)  # [batch_l * max_doc_l,75*3]
        elif (args.sent_attention == 'mean'):
            r = r * mask
            if name=="sent":
                r = tf.reduce_sum(r, 1) / tf.expand_dims(tf.cast(tf.reshape(self.sent_len, [-1]), tf.float32), 1)
            else:
                r = tf.reduce_sum(r, 1) / tf.expand_dims(tf.cast(tf.reshape(self.doc_len, [-1]), tf.float32), 1)
        elif (args.sent_attention == 'max'):
            r = r + (mask - 1) * 999
            r = tf.reduce_max(r, 1)

        return r

    def atten_score(self,name,mask1,mask2):#mask ok
        with tf.variable_scope(name,reuse=True):
            W_pc=tf.get_variable("w_pc")#ok
            W_r = tf.get_variable("w_r")

        if name=="sent":
            max_len=self.max_sent_len
            input_embed=self.token_str
            print(name,input_embed)

        else:
            input_embed = self.doc_str
            max_len=self.max_doc_len
            print(name,input_embed)

        tp=tf.layers.dense(tf.reshape(input_embed,[-1,2*args.emb_str]),2*args.emb_str,tf.nn.tanh)#not ok
        tc=tf.layers.dense(tf.reshape(input_embed,[-1,2*args.emb_str]),2*args.emb_str,tf.nn.tanh)
        tp=tf.reshape(tp,[-1,max_len,2*args.emb_str])#(B*T,S,2*50)
        tc=tf.reshape(tc,[-1,max_len,2*args.emb_str])
        tmp=tf.tensordot(tp,W_pc,[[-1],[0]])#equals to matmul at last 2 dim,(B*T,S,2*50)
        atten_score_words=tf.exp(tf.matmul(tmp,tc,transpose_b=True))#(B*T,S,S)
        atten_score_roots=tf.exp(tf.squeeze(tf.tensordot(input_embed,W_r,[-1,0])))#(B*T,S,)
        tmp=tf.zeros_like(atten_score_words[:,:,0])#(B*T,S,)
        A=tf.matrix_set_diag(atten_score_words,tmp)#Aij=0 if i=j,(B*T,S,S)

        proot,pz=self.get_matrix_tree(atten_score_roots,A,mask1,mask2)

        return proot,pz

    def get_matrix_tree(self,r,A,mask1,mask2):
        L=tf.zeros_like(A)
        L=L-A
        tmp=tf.reduce_sum(A,1)
        L=tf.matrix_set_diag(L,tmp)

        L_dash=tf.concat([L[:,1:,:],tf.expand_dims(r,1)],1)#(B*T,S,S)
        L_dash_inv=tf.matrix_inverse(L_dash)
        proot=tf.multiply(r,L_dash_inv[:,:,0])##(B*T,S,)
        pz1=mask1*tf.multiply(A,tf.matrix_transpose(tf.expand_dims(tf.matrix_diag_part(L_dash_inv),2)))#(B*T,S,S)
        pz2=mask2*tf.multiply(A,tf.matrix_transpose(L_dash_inv))#(B*T,S,S)
        pz=pz1-pz2

        return proot,pz

    def add_project(self):
        dim=self.doc_r.shape[1]
        output=tf.layers.dropout(self.doc_r,args.drop_rate,training=self.is_training)
        output=tf.layers.dense(output,dim,activation=tf.nn.relu)
        output=tf.layers.dropout(output,args.drop_rate,training=self.is_training)
        output=tf.layers.dense(output,dim,activation=tf.nn.relu)
        self.logits=tf.layers.dense(output,args.n_class)

    def add_backward(self):
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits))

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

    def train(self,X_train,y_train,X_val,y_val):
        total = len(X_train) // args.batch_size + 1
        for epoch in range(args.num_epochs):
            count=0
            for step, data in enumerate(self.gen_batch(X_train, y_train, args.batch_size)):
                feed_dict = {self.input: data["input"], self.doc_len: data["doc_len"],
                             self.labels: data["labels"], self.max_sent_len:data["max_sent_len"],self.is_training: True}
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f"
                      % (epoch + 1, args.num_epochs, step + 1, total, loss))

                if step % 10000 == 0 and step != 0:
                    print("validating......")
                    total_correct=tf.constant(0,dtype=tf.float32)
                    for data_val in self.gen_batch(X_val, y_val, args.batch_size):
                        feed_dict = {self.input: data_val["input"], self.doc_len: data_val["doc_len"],
                                     self.labels: data_val["labels"], self.max_sent_len:data_val["max_sent_len"],self.is_training: False}
                        correct=tf.reduce_sum(tf.cast(tf.equal(self.labels,tf.argmax(self.logits,1)),tf.float32))
                        total_correct=tf.add(total_correct,correct)
                        acc=self.sess.run(total_correct,feed_dict=feed_dict)
                        #print(acc)

                    print("val_acc: %.3f" %(acc/len(X_val)))


    #def predict(self):
    #    pass

    def gen_batch(self,X,y,batch_size):
        data={}
        data["label"],data["input"],data["doc_len"]=[],[],[]

        def pad(batch_x):
            doc_len=[]
            max_doc_len=max([len(doc) for doc in batch_x])
            max_sent_len=max([len(x) for doc in batch_x for x in doc])
            #max_doc_len=60
            #max_sent_len=150
            batch_x_pad=np.zeros([batch_size,max_doc_len,max_sent_len])
            for i,doc in enumerate(x_batch):
                doc_len.append(len(doc))
                for j,sent in enumerate(doc):
                    batch_x_pad[i,j,:len(sent)]=np.asarray(sent,dtype=int)
            return batch_x_pad,np.array(doc_len),max_sent_len

        for i in range(0,len(X)//batch_size+1):
            x_batch=X[i*batch_size:(i+1)*batch_size]
            data["labels"]=y[i*batch_size:(i+1)*batch_size]
            data["input"], data["doc_len"],data["max_sent_len"]=pad(x_batch)
            yield  data