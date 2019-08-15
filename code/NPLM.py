import time
import math
import numpy as np
import tensorflow as tf
from collections import Counter

flags = tf.flags
FLAGS = flags.FLAGS

# parameters 
flags.DEFINE_string('path_corpus', '../data/text8', 'the path of the corpus file')
flags.DEFINE_integer('train_batch_size', 120, 'number of words in a batch for train')
flags.DEFINE_integer('vocab_size', 10000, '词表中单词数的上限')
flags.DEFINE_integer('window_size', 5, 'n-gram(前n-1个词)')
flags.DEFINE_integer('num_epochs', 1, 'num_epochs')
flags.DEFINE_integer('embedding_dim', 50, '词向量的维度')
flags.DEFINE_integer('hidden_size', 100, '隐层神经元的个数')
flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')


# data pre-process
class CorpusProcess:
    def __init__(self, batch_size, seq_length, dict_size):
        self.batch_size = batch_size
        self.context_window_size = seq_length
        self.batch_ptr = 0
        self.vocab_size = dict_size
        self.dic_wrd = {}
        self.data = []
        self.reverse_dic_wrd = {}
        self.num_batches = 0
        self.pre_process(FLAGS.path_corpus)

    # data pre_processing
    def pre_process(self, input_file):

        with open(input_file, 'r') as f:
            str_ = f.read()
        words = str_.split()
        print('file read')

        count = [['UNK', -1]]
        frq_word = Counter(words)
        frq_word = frq_word.most_common(FLAGS.vocab_size - 1)
        count.extend(frq_word)

        for word, _ in count:
            self.dic_wrd[word] = len(self.dic_wrd)
        self.vocab_size = len(self.dic_wrd)
        print('counter words')
        for word in words:
            idx = 0
            if word in self.dic_wrd:
                idx = self.dic_wrd[word]
            self.data.append(idx)
        print('dict created')
        self.num_batches = int((len(self.data) - 1) / self.batch_size) + 1
        # words
        self.reverse_dic_wrd = dict(zip(self.dic_wrd.values(), self.dic_wrd.keys()))

    # to get next batch
    def next_batch(self): 
        start_idx = self.batch_ptr
        end_idx = min(self.batch_ptr + self.batch_size, len(self.data))
        batch_data = self.data[start_idx:end_idx]
        self.batch_ptr = (self.batch_ptr + 1) % self.num_batches
        x_data, y_data = [], []
        for i in range(len(batch_data) - self.context_window_size):
            x_data.append(batch_data[i:i + self.context_window_size])
            y_data.append([batch_data[i + self.context_window_size]])
        return x_data, y_data


# calling for pre_processing
corpus = CorpusProcess(FLAGS.train_batch_size, FLAGS.window_size, FLAGS.vocab_size)
vocab_size = corpus.vocab_size

# building the computational graph for nplm
graph = tf.Graph()
with graph.as_default():
    # creating placeholder
    input_data = tf.compat.v1.placeholder(tf.int32, [FLAGS.train_batch_size-FLAGS.window_size, FLAGS.window_size])
    targets = tf.compat.v1.placeholder(tf.int64, [FLAGS.train_batch_size-FLAGS.window_size, 1])

    # creating the variable scope for embeds
    with tf.compat.v1.variable_scope('nplm_embed'):
        embeds = tf.Variable(tf.random.uniform([vocab_size, FLAGS.embedding_dim], -1.0, 1.0))
        embeds = tf.nn.l2_normalize(embeds, 1)

    with tf.compat.v1.variable_scope('nplm_weight'):
        # weight matrix variable for word embedding to hidden layer
        # this variable is initialized with random values
        weight_h = tf.Variable(tf.random.truncated_normal([FLAGS.window_size * FLAGS.embedding_dim + 1, FLAGS.hidden_size],
                                                          stddev=1.0 / math.sqrt(FLAGS.hidden_size)))
        # softmax variable
        softmax_w = tf.Variable(tf.random.truncated_normal([FLAGS.window_size * FLAGS.embedding_dim, vocab_size],
                                                           stddev=1.0 / math.sqrt(FLAGS.window_size * FLAGS.embedding_dim)))
        softmax_u = tf.Variable(tf.random.truncated_normal([FLAGS.hidden_size + 1, vocab_size],
                                                           stddev=1.0 / math.sqrt(FLAGS.hidden_size)))
    # lookup of nn  embedding and the with data input data
    in_emb = tf.nn.embedding_lookup(embeds, input_data)
    in_emb = tf.reshape(in_emb, [-1, FLAGS.window_size * FLAGS.embedding_dim])
    # add bias item
    tf_ones = tf.ones([tf.shape(input_data)[0], 1])
    in_emb_add = tf.concat([in_emb, tf_ones], 1)
    # tanh activation function
    inputs = tf.tanh(tf.matmul(in_emb_add, weight_h))
    inputs_add = tf.concat([inputs, tf.ones([FLAGS.train_batch_size-FLAGS.window_size, 1])], 1)
    # matmul for output
    opts = tf.matmul(inputs_add, softmax_u) + tf.matmul(in_emb, softmax_w)
    # loss function and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=opts, labels=tf.squeeze(targets)))
    optimizer = tf.compat.v1.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)

    # normalized embedding
    embeds_norm = tf.sqrt(tf.reduce_sum(tf.square(embeds), 1, keepdims=True))
    norm_embed = embeds / embeds_norm


# running the graph in the session
with tf.compat.v1.Session(graph=graph) as sess:
    # initializing global variables
    tf.compat.v1.global_variables_initializer().run()
    for num_epoch in range(FLAGS.num_epochs):
        for b in range(corpus.num_batches):
            start = time.time()
            # get next batch
            x, y = corpus.next_batch()
            feed = {input_data: x, targets: y}
            # training for each batch
            train_loss,  _ = sess.run([loss, optimizer], feed)
            end = time.time()
            if b % 10 == 0 or b == corpus.num_batches - 1:
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, corpus.num_batches, num_epoch + 1, train_loss, end - start))
    # word embedding
    np.save('wrd_embeds', norm_embed.eval())

