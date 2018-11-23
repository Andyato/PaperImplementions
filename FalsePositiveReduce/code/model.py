import tensorflow as tf
import time

class model():
    def __init__(self, name, _input, f_sizes, pool_ksize, fc_size, keep_prob):
        self.name = name
        self._input = _input
        self.filters = f_sizes
        self.strides = [1, 1,1,1, 1]
        self.ksize = pool_ksize
        self.fc1_size = fc_size[0]
        self.fc2_size = fc_size[1]
        self.keep_prob = keep_prob
        
        self.cubic_shape = [[ 6, 20, 20], 
                            [10, 30, 30], 
                            [26, 40, 40]]

    def build(self):
        with tf.name_scope(self.name):
            conv1 = conv3d(self._input, self.filters[0], self.strides, name='conv1')
            pool1 = max_pool(conv1, self.strides, self.ksize)
            conv2 = conv3d(pool1, self.filters[1], self.strides, name="conv2")
            conv3 = conv3d(conv2, self.filters[2], self.strides, name="conv3")
            fc_1  = fc(conv3, self.fc1_size, "fc1")
            drop  = dropout(fc_1, self.keep_prob)
            fc_2  = fc(drop, self.fc2_size, "fc2")

            return fc_2

    def inference(self, X, epoch, momentum=0.9, train=True):
        model_index = int(self.name[-1]) - 1
        cub_shape = self.cubic_shape[model_index]
        keep_prob = tf.placeholder(tf.float32)
        X = tf.placeholder(tf.float32, [None, cub_shape[0], cub_shape[1], cub_shape[2]])
        logits = self.build()

        saver = tf.train.Saver()

        if train:
            labels = tf.placeholder(tf.float32, [None, 2])
            lr = tf.placeholder(tf.float32, 0.3)

            loss_ = loss(logits, labels)

            train_step = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_)

            accuracy_ = accuracy(logits, labels)

            merged = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                Writer = tf.summary.FileWriter('./tensorboard/', sess.graph)

                for i in range(epoch):
                    epoch_start =time.time()
                    for _ in range(steps):
                        # TO DO 
                        train_batch, train_label = get_train_batch()
                        feet_dict = {X:train_batch, labels:train_label, keep_prob: self.keep_prob}
                        _,summary = sess.run([train_step, merged],feed_dict =feed_dict)
                        Writer.add_summary(summary, i)
                        saver.save(sess, './ckpt/{}/'.format(self.name), global_step=i + 1)

                    epoch_end = time.time()
                    # TO DO
                    test_batch,test_label = get_test_batch(test_path)
                    test_dict = {X: test_batch, labels: test_label, keep_prob:1.0}
                    test_acc, test_loss = sess.run([accruacy_, loss_],feed_dict=test_dict)

                    print("Test loss: {:.2f}, test accuracy: {:.2f}".format(test_acc, test_loss))
                    print("Epoch %d , time consumed %.2f seconds"%(i,(epoch_end-epoch_start)))


def get_weights(shape, name):
    with tf.name_scope(name) as scope:
        return tf.Variable(tf.random_normal(shape=shape, stddev=0.001), dtype=tf.float32, name=scope.name)

def get_bias(shape, name):
    with tf.name_scope(name) as scope:
        return tf.Variable(tf.constant(0.001, shape=shape),dtype=tf.float32, name=scope.name)

def conv3d(_input, f_size, strides, name, padding="VALID"):
    with tf.name_scope(name) as scope:
        w = get_weights(f_size, name=scope.name+"_weights")
        b = get_bias(f_size[-1], name=scope.name+"_bias")
        conv = tf.nn.conv3d(_input, w, strides, padding=padding)
        relu = tf.nn.relu( tf.add(conv, b) )
        return relu

def max_pool(_input, strides, ksize, padding="SAME"):
    with tf.name_scope("max_pool"):
        return tf.nn.max_pool3d(input=_input, strides=strides, ksize=kszie, padding=padding)

def fc(_input, out_dim, name):
    with tf.name_scope(name) as scope:
        shape = _input.get_shape().as_list()
        in_dim = 1
        for i in shape[1:]:
            in_dim *= i
        x = tf.reshape(_input, [-1, in_dim])
        w = get_weights([in_dim, out_dim], name=scope.name+"_weights")
        b = get_bias([out_dim], scope.name+"_bias")
        relu = tf.nn.relu( tf.add(tf.matmul(w,x), b) )

        return relu

def dropout(_input, keep_prob):
    return tf.nn.dropout(_input, keep_prob)

def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss

def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy', accuracy)
        return accuracy
