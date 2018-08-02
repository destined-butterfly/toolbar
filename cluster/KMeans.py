import tensorflow as tf

class KMeans(object):

    def __init__(self, n_cluster=3, init='default', max_iter = 300, tol=0.0001, random_state = 0):
        self.n_cluster = n_cluster
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.data_train = None
        self.num_features = None
        self.num_samples = None
        self.x = None
        self.init_centers_ = None
        self.reshape_data = None
        self.reshape_center_cluster = None
        self.center = None
        self.distance_ = None
        self.y = None
        self.new_center = None
        self.sse_ = None
        self.inertia_ = None
        self.sess = None

    def clean(self):
        self.data_train = None
        self.num_features = None
        self.num_samples = None
        self.x = None
        self.init_centers_ = None
        self.reshape_data = None
        self.reshape_center_cluster = None
        self.center = None
        self.distance_ = None
        self.y = None
        self.new_center = None
        self.sse_ = None
        self.inertia_ = None
        self.sess = None

    def center_init(self):
        """return the self.init_centers_"""
        if self.init == 'default':
            self.init_centers_ = tf.Variable(tf.slice(self.x,
                                                      [0, 0], [self.n_cluster, self.num_features]))
        return self.init_centers_

    def reshape_x(self):
        """tile and reshape N*k*self.num_features"""
        self.reshape_data = tf.reshape(tf.tile(self.x, [1, self.n_cluster]), [self.num_samples, self.n_cluster,
                                                                              self.num_features])

    def reshape_center(self):
        self.reshape_center_cluster = tf.reshape(tf.tile(self.center, [self.num_samples, 1]),
                                                 [self.num_samples, self.n_cluster, self.num_features])

    def distance(self):
        self.distance_ = tf.reduce_sum(tf.square(self.reshape_center_cluster - self.reshape_data), reduction_indices=2)

    def update_center(self):
        sum_ = tf.unsorted_segment_sum(self.x, self.y, self.n_cluster)
        tol = tf.unsorted_segment_sum(tf.ones_like(self.x), self.y, self.n_cluster)
        self.new_center = tf.Variable(sum_ / tol)

    def sse(self):
        """w:one_hot"""
        one_hot = tf.one_hot(self.y, self.n_cluster)
        """reshape center of cluster"""
        self.reshape_center()
        """see"""
        self.distance()
        self.sse_ = tf.reduce_sum(self.distance_ * one_hot)
        return self.sse_

    def set_graph(self):
        """data to train"""
        self.x = tf.placeholder("float", [None, self.num_features])
        """init some plots as center of cluster"""
        self.center = self.center_init()
        """reshape the data to train : self.reshape_data"""
        self.reshape_x()
        """reshape the center of cluster"""
        self.reshape_center()
        """cluster"""
        self.distance()
        self.y = tf.argmin(self.distance_, 1)
        """ the new center of cluster"""
        self.update_center()

    def fit_predict(self, data_train):
        """fit the model """
        """"""
        self.clean()
        """"""
        self.data_train = data_train
        """"""
        self.num_features = int(tf.Session().run(tf.shape(self.data_train[1])))
        self.num_samples = int(tf.Session().run(tf.shape(self.data_train[0])))
        """"""
        if self.num_samples >= self.n_cluster:
            """set graph"""
            self.set_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            """max_iter"""
            i = 0
            diff = True
            while i < self.max_iter and diff:
                new_center = self.sess.run(self.new_center, feed_dict={self.x: self.data_train})
                diff = tf.reduce_any(tf.not_equal(new_center, self.center))
                if diff and i < self.max_iter - 1:
                    self.center = new_center
                i += 1
            print self.sess.run(self.y, feed_dict={self.x: self.data_train})
            """sse"""
            self.inertia_ = self.sess.run(self.sse_, feed_dict={self.x: self.data_train})
        else:
            print "failure: number of samples < number of cluster"

    def predict(self, data):
        #self.num_features = int(tf.Session().run(tf.shape(data[1])))
        self.num_samples = int(tf.Session().run(tf.shape(data[0])))
        print self.sess.run(self.y, feed_dict={self.x: data})

