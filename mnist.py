# encoding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class Mnistrec(object):

    __doc__ = """
    Gray Scala Image
    fit(): train the model
    predict(): predict labels
    """

    def __init__(self, n_iter=2000):
        """迭代次数"""
        self.n_iter = n_iter
        """训练新的model前需要clean参数"""
        self.x = None
        self.y_ = None
        self.w = None
        self.b = None
        self.y = None
        self.cross_entropy = None
        self.train_step = None
        self.n_feature = None
        self.n_y_hot = None
        self.correct_prediction = None
        self.accuracy = None
        self.y_value = None
        self.sess = None
        """if完成fit"""
        self.finish_fit = False

    def set_graph(self):
        """构建整个计算图"""
        """待输入的数据:x，y占位符"""
        self.x = tf.placeholder("float", [None, self.n_feature])
        self.y_ = tf.placeholder("float", [None, self.n_y_hot])
        """权重和偏置变量（优化参数）"""
        self.w = tf.Variable(tf.zeros([self.n_feature, self.n_y_hot]))
        self.b = tf.Variable(tf.zeros([self.n_y_hot]))
        """目标y"""
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        """模型评估交叉熵"""
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        """训练模型和优化参数，使用梯度下降算法，最小化交叉熵"""
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        """正确预测和准确率"""
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        """预测值"""
        self.y_value = tf.argmax(self.y, 1)

    def fit(self, data):
        self.clean()
        """获取训练数据x和y的维度"""
        self.n_feature = int(tf.Session().run(tf.shape(data.train.images)[1]))
        self.n_y_hot = int(tf.Session().run(tf.shape(data.train.labels)[1]))
        """根据输入的待训练数据构建图"""
        self.set_graph()
        """训练模型"""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.n_iter):
            batch = data.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1]})
        print "训练完成"
        """if完成fit"""
        self.finish_fit = True

    def predict(self, data_x):
        print "预测值为:",self.sess.run(self.y_value, feed_dict={self.x: data_x})

    def predict_and_accuracy(self, data_x, data_y):
        print "正确率为:%.2f%%" %(self.sess.run(self.accuracy*100, feed_dict={self.x: data_x, self.y_: data_y}))

    def clean(self):
        self.x = None
        self.y_ = None
        self.w = None
        self.b = None
        self.y = None
        self.cross_entropy = None
        self.train_step = None
        self.n_feature = None
        self.n_y_hot = None
        self.correct_prediction = None
        self.accuracy = None
        self.finish_fit = False
        self.y_value = None
        self.sess = None

data = input_data.read_data_sets("MNIST_data/", one_hot=True)
model = Mnistrec()
model.fit(data)
model.predict(data_x= data.test.images[:4])
print "真实值为:", tf.Session().run(tf.argmax(data.test.labels[:4], 1))
model.predict_and_accuracy(data.test.images, data.test.labels)



