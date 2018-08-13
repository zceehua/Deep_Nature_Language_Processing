import tensorflow as tf
from config import args
from tools import eval_confusion_matrix
import numpy as np
#_no_value = object()

PRETRAIN_PATH="./model/pretrain"

class Model(object):
    def __init__(self,logger,vocab_size,embedding=None):
        self.logger=logger
        self.vocab_size=vocab_size
        self.embedding=embedding
        self.lr2 = np.array([0.00015, 0.0004, 0.001, 0.002])


    def get_embedding(self,input,reuse=False):
        with tf.variable_scope("embedding",reuse=reuse):
            if self.embedding is None:
                embedding = tf.get_variable("embedding_matrix",[self.vocab_size, args.embedding_size], tf.float32)
            else:
                embedding = tf.Variable(initial_value=self.embedding,
                                             name="embedding_matrix",
                                             dtype="float32")
            embed = tf.nn.embedding_lookup(embedding, input)
            cnn_inputs = tf.expand_dims(embed, -1)#(B,W,H,1)
            # Input dropout
            cnn_inputs = tf.nn.dropout(cnn_inputs, keep_prob=args.dropout)
        return  cnn_inputs

    def lstm_cell(self,rnn_size):
        return tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.orthogonal_initializer())

    def cnn_module(self,input,reuse=False):
        with tf.variable_scope("cnn_module",reuse=reuse):
            cnn_input=self.get_embedding(input)
            filter_sizes=args.filter_sizes.split(",")
            filter_sizes=[int(x) for x in filter_sizes]
            max_feat_len=args.max_len-max(filter_sizes)+1
            #total_filters = args.num_filters * len(filter_sizes)
            output=[]
            for i, kernel_w in enumerate(filter_sizes):
                kernel = [kernel_w, args.embedding_size]
                # [B,seq-kw+1,1,num_filters]
                conv_out = tf.layers.conv2d(inputs=cnn_input, filters=args.num_filters,
                                            padding="valid", kernel_size=kernel, activation=tf.nn.relu)
                if args.clf=="cnn":
                    pool_out = tf.layers.max_pooling2d(inputs=conv_out, pool_size=[args.max_len - kernel_w + 1, 1],
                                                       strides=1)
                    pool_out=tf.squeeze(pool_out)
                    output.append(pool_out)
                elif args.clf=="clstm":
                    conv_out=tf.squeeze(conv_out,[2])
                    conv_out=conv_out[:,:max_feat_len,:]
                    output.append(conv_out)
            if len(filter_sizes)>1:
                cnn_out=tf.concat(output,-1)
            else:
                cnn_out=output
            pad = tf.zeros([args.batch_size,args.max_len-max_feat_len, len(filter_sizes)*args.num_filters])
            cnn_out=tf.concat([cnn_out,pad],1)
        return cnn_out

    def lstm_module(self,input,reuse=False):
        with tf.variable_scope("lstm_module", reuse=reuse):
            cells=[self.lstm_cell(args.hidden_size) for _ in range(args.nlayers)]
            cells=[tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=args.dropout) for cell in cells]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.initial_state = cells.zero_state(args.batch_size, tf.float32)
            output, state = tf.nn.dynamic_rnn(cells, input, self.seq_len, self.initial_state)
        return output,state[-1].h

    def forward(self,features,mode):
        answer,question=features["answer"],features["question"]
        self.seq_len=self.get_seq_len(answer,question)
        if args.clf=="cnn":
            cnn_ans = self.cnn_module(answer)
            cnn_ques = self.cnn_module(question, reuse=True)
            cnn_output = tf.concat([cnn_ans, cnn_ques], -1)
            hidden=cnn_output
        elif args.clf=="lstm":
            ans_emb=self.get_embedding(answer)
            ques_emb=self.get_embedding(question,reuse=True)
            lstm_input=tf.concat([ans_emb,ques_emb],-1)
            _,hidden=self.lstm_module(lstm_input)
        elif args.clf=="clstm":
            cnn_ans = self.cnn_module(answer)
            cnn_ques = self.cnn_module(question, reuse=True)
            cnn_output = tf.concat([cnn_ans, cnn_ques], -1)
            rnn_output,hidden=self.lstm_module(cnn_output)
        else:
            raise ValueError('clf should be one of [cnn, lstm, clstm]')

        logits,predictions=self.softmax_module(rnn_output,hidden)
        return rnn_output,logits,predictions

    def model_fn(self,features,labels,mode,params):
        rnn_output,logits, predictions=self.forward(features,mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss, train_op = self.loss_module(rnn_output,logits, labels)

        if  args.load_model:
            exclude = ['ft_softmax/',"global_step"]
            self.load_model(exclude=exclude)
            #self.load_model(include=["pt_softmax/"])
            args.load_model=False
            #print(args.load_model)

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=labels,predictions=predictions,name='acc_op')
            confusion_matrix = eval_confusion_matrix(labels, predictions, args.num_class)
            metrics = {'accuracy': accuracy, "confusion_matrix": confusion_matrix}
            tf.summary.scalar('accuracy', accuracy[1])
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.TRAIN:
           if args.fine_tune:
               logging_hook = tf.train.LoggingTensorHook({'global_step': self.global_step,
                                                          "lr":self.learning_rate_exp,
                                                          "lr_1":self.lr_var[0],"lr_2":self.lr_var[1],
                                                          "lr_3": self.lr_var[2],"lr_4":self.lr_var[3]}, every_n_iter=100)
           else:
               logging_hook = tf.train.LoggingTensorHook({'global_step': self.global_step,
                                                          "lr": self.learning_rate_exp},
                                                         every_n_iter=100)
           return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


    def softmax_module(self,output,hidden,reuse=False):
        if not args.pretrain:
            #do specifiy the name of layer if we want to get kernel of the layer
            with tf.variable_scope("ft_softmax", reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
                logits=tf.layers.dense(hidden,args.num_class,name="dense")
        else :
            with tf.variable_scope("pt_softmax", reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
                output=tf.reshape(output,[-1,args.hidden_size])
                logits=tf.layers.dense(output,self.vocab_size,name="dense")

        predictions=tf.argmax(tf.nn.softmax(logits),-1)
        return logits,predictions


    def sequence_loss(self,logits,labels):
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=tf.reshape(logits, [args.batch_size, args.max_len, self.vocab_size]),
            targets=labels,
            weights=tf.ones([args.batch_size, args.max_len]),
            average_across_timesteps=True,
            average_across_batch=True,
        )
        return loss

    #increase training speed when output class is large
    def sampled_loss(self,logits,labels):
        #discard <PAD>
        mask = tf.reshape(tf.to_float(tf.sign(labels)), [-1])
        with tf.variable_scope('pt_softmax/dense', reuse=True):
            _weights = tf.transpose(tf.get_variable('kernel'))
            _biases = tf.get_variable('bias')
        loss = tf.reduce_sum(mask * tf.nn.sampled_softmax_loss(
            weights=_weights,
            biases=_biases,
            labels=tf.reshape(labels, [-1, 1]),#[batch_size, num_true]
            inputs=tf.reshape(logits, [-1, args.hidden_size]),
            num_sampled=args.num_sampled,
            num_classes=self.vocab_size,
        )) / tf.to_float(tf.shape(logits)[0])
        return loss

    def loss_module(self,rnn_output,logits,labels):
        self.global_step = tf.train.get_global_step()
        if not args.pretrain:
            loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        else:
            if args.num_sampled>self.vocab_size:
                loss=self.sequence_loss(logits,labels)
            else:
                loss=self.sampled_loss(rnn_output,labels)
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_losses
        params = tf.trainable_variables()
        self.learning_rate_exp = tf.train.exponential_decay(args.lr, self.global_step,
                                                            args.decay_size,
                                                            args.decay_factor)
        if args.fine_tune:
            cnn_vars,other_vars,sm_vars=self.get_vars(params)
            grad_op=self.get_grad_op(cnn_vars,other_vars,sm_vars,loss,params)
        else:
            optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.grad_clip)
            grad_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        var_avg = tf.train.ExponentialMovingAverage(args.moving_average_decay,self.global_step)
        variables_averages_op = var_avg.apply(params)
        with tf.control_dependencies([grad_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return loss,train_op

    #Discriminative fine-tuning
    def get_grad_op(self,cnn,other,sm,loss,params):
        lstm_ops=[]
        lstm_grads=[]
        #self.lr2=self.stlr(self.lr2)
        self.lr_var={}
        for i in range(len(self.lr2)):
            self.lr_var[i]=self.stlr(self.lr2[i])
        cnn_op = tf.train.AdamOptimizer(self.learning_rate_exp)
        sm_op = tf.train.AdamOptimizer(self.stlr(self.lr_var[3]))
        for i in range(len(other)):
            lstm_ops.append(tf.train.AdamOptimizer(self.stlr(self.lr_var[i])))
        grads = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(grads, clip_norm=args.grad_clip)
        grad_cnn=grads[:len(cnn)]
        offset=len(cnn)
        for i in range(len(other)):
            lstm_grads.append(grads[offset:offset+len(other[i])])
            offset=offset+len(other[i])
        grad_sm=grads[offset:]
        cnn_train=cnn_op.apply_gradients(zip(grad_cnn,cnn), global_step=self.global_step)
        sm_train=sm_op.apply_gradients(zip(grad_sm,sm), global_step=self.global_step)
        lstm_train=[lstm_ops[i].apply_gradients(zip(lstm_grads[i],other[i]), global_step=self.global_step) for i in range(len(other))]
        train_op=tf.group(cnn_train,lstm_train[0],lstm_train[1],lstm_train[2],sm_train)
        return train_op

    #Slanted triangular learning rates
    def stlr(self,eta_max=0.01):
        T=args.num_steps*args.n_epochs*args.amount
        t=tf.to_float(self.global_step + 1)#prevent 0 steps
        cut_frac=0.1
        ratio=32
        cut=tf.to_float(T*cut_frac)
        f1=lambda :tf.subtract(tf.to_float(1),tf.subtract(t,cut)/(tf.multiply(cut,tf.to_float(1/cut_frac - 1))))
        f2=lambda :tf.divide(t,cut)
        p=tf.cond(tf.greater(cut,t),f2,f1)
        eta_t=tf.multiply(tf.to_float(eta_max),tf.divide(tf.add(tf.to_float(1),tf.multiply(p,tf.to_float(ratio-1))),tf.to_float(ratio)))
        eta_t=tf.maximum(tf.to_float(0.000001),eta_t)#prevent negative
        return eta_t


    def get_vars(self,params):
        cnn_vars = [var for var in params if "cnn_module" in var.name]
        other_vars = []#trainable parameters for lstm layers
        lstm_layer = "lstm_module/rnn/multi_rnn_cell/cell_"
        for i in range(args.nlayers):
            vars = [var for var in params if lstm_layer + str(i) in var.name]
            other_vars.append(vars)
        sm_vars = [var for var in params if "pt_softmax" in var.name]#trainable parameters for softmax layers
        #other_vars.append(vars)
        return cnn_vars,other_vars,sm_vars

    def load_model(self,include=None,exclude=None):
        self.logger.info("loading pretrained model ..")
        #exclude = ['pt_softmax/', 'cnn_module_1/embedding/','cnn_module/embedding/','ft_softmax/']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude,include=include)
        tf.train.init_from_checkpoint(PRETRAIN_PATH,
                                      {v.name.split(':')[0]: v for v in variables_to_restore})
        #[<tf.Variable 'global_step:0' shape=() dtype=int64_ref>, <tf.Variable 'cnn_module/conv2d/kernel:0' shape=(3, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/kernel:0' shape=(4, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/kernel:0' shape=(5, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(896, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'beta1_power:0' shape=() dtype=float32_ref>, <tf.Variable 'beta2_power:0' shape=() dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/kernel/Adam:0' shape=(3, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/kernel/Adam_1:0' shape=(3, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/bias/Adam:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/bias/Adam_1:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/kernel/Adam:0' shape=(4, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/kernel/Adam_1:0' shape=(4, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/bias/Adam:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/bias/Adam_1:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/kernel/Adam:0' shape=(5, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/kernel/Adam_1:0' shape=(5, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/bias/Adam:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/bias/Adam_1:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam:0' shape=(896, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adam_1:0' shape=(896, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adam_1:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adam_1:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adam_1:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adam:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adam_1:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adam:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adam_1:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/kernel/ExponentialMovingAverage:0' shape=(3, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/bias/ExponentialMovingAverage:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/kernel/ExponentialMovingAverage:0' shape=(4, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/bias/ExponentialMovingAverage:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/kernel/ExponentialMovingAverage:0' shape=(5, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/bias/ExponentialMovingAverage:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ExponentialMovingAverage:0' shape=(896, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ExponentialMovingAverage:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ExponentialMovingAverage:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ExponentialMovingAverage:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/ExponentialMovingAverage:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/ExponentialMovingAverage:0' shape=(512,) dtype=float32_ref>]
        #print(variables_to_restore)
        #[<tf.Variable 'cnn_module/embedding/embedding_matrix:0' shape=(2085, 200) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/kernel:0' shape=(3, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/kernel:0' shape=(4, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_1/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/kernel:0' shape=(5, 200, 1, 128) dtype=float32_ref>, <tf.Variable 'cnn_module/conv2d_2/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'cnn_module_1/embedding/embedding_matrix:0' shape=(2085, 200) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(896, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:0' shape=(256, 512) dtype=float32_ref>, <tf.Variable 'lstm_module/rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'ft_softmax/dense/kernel:0' shape=(128, 4) dtype=float32_ref>, <tf.Variable 'ft_softmax/dense/bias:0' shape=(4,) dtype=float32_ref>]
        #print(tf.trainable_variables())

    def get_seq_len(self,ans,ques):
        ans_len=tf.count_nonzero(ans,-1)
        ques_len=tf.count_nonzero(ques,-1)
        cmp=tf.greater(ans_len,ques_len)
        seq_len=tf.where(cmp,ans_len,ques_len)
        return seq_len