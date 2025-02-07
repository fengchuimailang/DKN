import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

tf.compat.v1.disable_eager_execution()  # 更换 tf2  需要处理兼容问题

class DKN(object):
    def __init__(self, args):
        self.params = []  # for computing regularization loss
        self._build_inputs(args)
        self._build_model(args)
        self._build_train(args)

    def _build_inputs(self, args):
        with tf.compat.v1.name_scope('input'):
            self.clicked_words = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_words')
            self.clicked_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_entities')
            self.news_words = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='news_words')
            self.news_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='news_entities')
            self.labels = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, args):
        with tf.compat.v1.name_scope('embedding'):
            word_embs = np.load('../data/news/word_embeddings_' + str(args.word_dim) + '.npy')
            entity_embs = np.load('../data/kg/entity_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')
            # 加载两种embedding
            self.word_embeddings = tf.Variable(word_embs, dtype=np.float32, name='word')
            self.entity_embeddings = tf.Variable(entity_embs, dtype=np.float32, name='entity')
            self.params.append(self.word_embeddings)
            self.params.append(self.entity_embeddings)

            if args.use_context:
                context_embs = np.load(
                    '../data/kg/context_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')
                self.context_embeddings = tf.Variable(context_embs, dtype=np.float32, name='context')
                self.params.append(self.context_embeddings)

            if args.transform:
                self.entity_embeddings = tf.compat.v1.layers.dense(
                    self.entity_embeddings, units=args.entity_dim, activation=tf.nn.tanh, name='transformed_entity',
                    kernel_regularizer=tf.keras.regularizers.l2(0.5 * (args.l2_weight)))
                if args.use_context:
                    self.context_embeddings = tf.compat.v1.layers.dense(
                        self.context_embeddings, units=args.entity_dim, activation=tf.nn.tanh,
                        name='transformed_context', kernel_regularizer=tf.keras.regularizers.l2(0.5 * (args.l2_weight)))

        user_embeddings, news_embeddings = self._attention(args)
        self.scores_unnormalized = tf.reduce_sum(input_tensor=user_embeddings * news_embeddings, axis=1)
        self.scores = tf.sigmoid(self.scores_unnormalized)

    def _attention(self, args):
        # 注意力机制
        # (batch_size * max_click_history, max_title_length)
        clicked_words = tf.reshape(self.clicked_words, shape=[-1, args.max_title_length])
        clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, args.max_title_length])

        with tf.compat.v1.variable_scope('kcnn', reuse=tf.compat.v1.AUTO_REUSE):  # reuse the variables of KCNN
            # (batch_size * max_click_history, title_embedding_length)
            # title_embedding_length = n_filters_for_each_size * n_filter_sizes
            clicked_embeddings = self._kcnn(clicked_words, clicked_entities, args)

            # (batch_size, title_embedding_length)
            news_embeddings = self._kcnn(self.news_words, self.news_entities, args)

        # (batch_size, max_click_history, title_embedding_length)
        clicked_embeddings = tf.reshape(
            clicked_embeddings, shape=[-1, args.max_click_history, args.n_filters * len(args.filter_sizes)])

        # (batch_size, 1, title_embedding_length)
        news_embeddings_expanded = tf.expand_dims(news_embeddings, 1)

        # (batch_size, max_click_history)
        attention_weights = tf.reduce_sum(input_tensor=clicked_embeddings * news_embeddings_expanded, axis=-1)

        # (batch_size, max_click_history)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        # (batch_size, max_click_history, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # (batch_size, title_embedding_length)
        user_embeddings = tf.reduce_sum(input_tensor=clicked_embeddings * attention_weights_expanded, axis=1)

        return user_embeddings, news_embeddings

    def _kcnn(self, words, entities, args):
        """
        kcnn 模块
        :param words:
        :param entities:
        :param args:
        :return:
        """
        # (batch_size * max_click_history, max_title_length, word_dim) for users
        # (batch_size, max_title_length, word_dim) for news
        embedded_words = tf.nn.embedding_lookup(params=self.word_embeddings, ids=words)
        embedded_entities = tf.nn.embedding_lookup(params=self.entity_embeddings, ids=entities)

        # (batch_size * max_click_history, max_title_length, full_dim) for users
        # (batch_size, max_title_length, full_dim) for news
        if args.use_context:
            embedded_contexts = tf.nn.embedding_lookup(params=self.context_embeddings, ids=entities)
            concat_input = tf.concat([embedded_words, embedded_entities, embedded_contexts], axis=-1)
            full_dim = args.word_dim + args.entity_dim * 2
        else:
            concat_input = tf.concat([embedded_words, embedded_entities], axis=-1)
            full_dim = args.word_dim + args.entity_dim

        # (batch_size * max_click_history, max_title_length, full_dim, 1) for users
        # (batch_size, max_title_length, full_dim, 1) for news
        concat_input = tf.expand_dims(concat_input, -1)

        outputs = []
        for filter_size in args.filter_sizes:
            filter_shape = [filter_size, full_dim, 1, args.n_filters]
            w = tf.compat.v1.get_variable(name='w_' + str(filter_size), shape=filter_shape, dtype=tf.float32)
            b = tf.compat.v1.get_variable(name='b_' + str(filter_size), shape=[args.n_filters], dtype=tf.float32)
            if w not in self.params:
                self.params.append(w)

            # (batch_size * max_click_history, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for users
            # (batch_size, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for news
            conv = tf.nn.conv2d(input=concat_input, filters=w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # (batch_size * max_click_history, 1, 1, n_filters_for_each_size) for users
            # (batch_size, 1, 1, n_filters_for_each_size) for news
            pool = tf.nn.max_pool2d(input=relu, ksize=[1, args.max_title_length - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1], padding='VALID', name='pool')
            outputs.append(pool)

        # (batch_size * max_click_history, 1, 1, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, 1, 1, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.concat(outputs, axis=-1)

        # (batch_size * max_click_history, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.reshape(output, [-1, args.n_filters * len(args.filter_sizes)])

        return output

    def _build_train(self, args):
        with tf.compat.v1.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores_unnormalized))
            self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                self.l2_loss = tf.add(self.l2_loss, args.l2_weight * tf.nn.l2_loss(param))
            if args.transform:
                self.l2_loss = tf.add(self.l2_loss, tf.compat.v1.losses.get_regularization_loss())
            self.loss = self.base_loss + self.l2_loss
            self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run(self.optimizer, feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        return auc
