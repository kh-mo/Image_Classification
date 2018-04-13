import os
import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

def Train_generator(path):
    image = list()
    label = list()
    folders = os.listdir(path)
    for folder in folders:
        subpath = path + "/" + folder
        for file in os.listdir(subpath):
            subfile = subpath + "/" + file
            image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
            label_tmp = len(folders)*[0]
            for position in range(len(folders)):
                if folder == folders[position]:
                    label_tmp[position] = 1
                    label.append(label_tmp)
    image = np.stack(image, axis=0)
    label = np.stack(label, axis=0)
    return (image, label)

def Test_generator(path):
    image = list()
    for folder in os.listdir(path):
        subfile = path + "/" + folder
        image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
    image = np.stack(image, axis=0)
    return image

class Alexnet():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        self.conv1 = self.Conv(input=self.x, filter=[11,11,3], output_channel=96, stride=4, init="he", name="conv1")
        self.act1 = self.Act(input=self.conv1, function="relu", name="relu1")
        self.pool1 = self.Pool(input=self.act1, filter=2, stride=2, function="max", name="pool1")

        self.conv2 = self.Conv(input=self.pool1, filter=[5,5,96], output_channel=256, stride=1, init="he", name="conv2")
        self.act2 = self.Act(input=self.conv2, function="relu", name="relu2")
        self.pool2 = self.Pool(input=self.act2, filter=2, stride=2, function="max", name="pool2")

        self.conv3 = self.Conv(input=self.pool2, filter=[3,3,256], output_channel=384, stride=1, init="he", name="conv3")
        self.act3 = self.Act(input=self.conv3, function="relu", name="relu3")

        self.conv4 = self.Conv(input=self.act3, filter=[3,3,384], output_channel=384, stride=1, init="he", name="conv4")
        self.act4 = self.Act(input=self.conv4, function="relu", name="relu4")

        self.conv5 = self.Conv(input=self.act4, filter=[3,3,384], output_channel=256, stride=1, init="he", name="conv5")
        self.act5 = self.Act(input=self.conv5, function="relu", name="relu5")
        self.pool3 = self.Pool(input=self.act5, filter=2, stride=2, function="max", name="pool5")

        self.fc1 = self.FC(input=self.pool3, output_channel=4096, init="he", name="fc1")
        self.act6 = self.Act(input=self.fc1, function="relu", name="relu6")

        self.fc2 = self.FC(input=self.act6, output_channel=4096, init="he", name="fc2")
        self.act7 = self.Act(input=self.fc2, function="relu", name="relu7")

        self.fc3 = self.FC(input=self.act7, output_channel=2, init="he", name="fc3")
        self.probability = tf.nn.softmax(self.fc3)

        ## cost, accuracy, train 지정
        with tf.variable_scope("cost"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc3))
            tf.summary.scalar("cross_entropy", self.cross_entropy)

        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc3, 1), tf.argmax(self.y, 1)), dtype=tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.variable_scope("training"):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.merge = tf.summary.merge_all()

    def variable_initializer(self, init="", filter=[], output_channel=0, name=""):
        with tf.variable_scope(name):
            if name[-4:] == "bias":
                variable = tf.Variable(tf.constant(0.1, shape=[output_channel]))
            else:
                shape = filter + [output_channel]
                if init == "norm":
                    variable = tf.truncated_normal(shape, mean=0, stddev=1, name=name)
                elif init == "xavier":
                    variable = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
                elif init == "he":
                    variable = tf.get_variable(name, shape,
                                               initializer=tf.contrib.layers.variance_scaling_initializer())
            return variable

    def Conv(self, input, filter, output_channel, stride, init, name):
        with tf.variable_scope(name):
            filt = self.variable_initializer(init, filter, output_channel, name + "_filt")
            bias = self.variable_initializer(output_channel=output_channel, name=name + "_bias")
            conv = tf.nn.conv2d(input, filt, [1, stride, stride, 1], padding="SAME") + bias
            return conv

    def Act(self, input, function, name):
        with tf.variable_scope(name):
            if function == "relu":
                act = tf.nn.relu(input, name=name)
            return act

    def Pool(self, input, filter, stride, function, name):
        with tf.variable_scope(name):
            if function == "max":
                pool = tf.nn.max_pool(input, ksize=[1, filter, filter, 1], strides=[1, stride, stride, 1],
                                      padding="SAME")
            elif function == "avg":
                pool = tf.nn.avg_pool(input, ksize=[1, filter, filter, 1], strides=[1, stride, stride, 1],
                                      padding="SAME")
            return pool

    def FC(self, input, output_channel, init, name):
        with tf.variable_scope(name):
            shape = 1
            for i, j in enumerate(input.get_shape().as_list()):
                if i > 0:
                    shape *= j
            weight = self.variable_initializer(init, [shape], output_channel, name + "_filt")
            bias = self.variable_initializer(output_channel=output_channel, name=name + "_bias")
            flat = tf.reshape(input, [-1, shape])
            fc = tf.matmul(flat, weight) + bias
            return fc

    def feed_dict(self, x, y, batch_size):
        batch_idx = np.random.choice(x.shape[0], batch_size, False)
        xs, ys = x[batch_idx], y[batch_idx]
        return {self.x: xs, self.y: ys}

    def train(self, sess, data, log_path, iteration, batch):
        train_x, train_y = data
        train_writer = tf.summary.FileWriter(log_path, sess.graph)

        ## training 시작
        for i in range(iteration):
            _ = sess.run(self.train_step, self.feed_dict(train_x, train_y, batch))
            if i % 100 == 0:
                _, acc, loss, summary = sess.run([self.train_step, self.accuracy, self.cross_entropy, self.merge],
                                                 self.feed_dict(train_x, train_y, batch))
                train_writer.add_summary(summary, i)
                print("step : %d, train accuracy : %g, train loss : %g" % (i, acc, loss))

    def test(self, sess, data):
        prob = sess.run(self.probability, feed_dict={self.x:data})
        return prob

    def save(self, sess, path):
        return tf.train.Saver().save(sess, path)

    def restore(self, sess, path):
        return tf.train.Saver().restore(sess, path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-train_path', type=str, default="C:/Users/user/Desktop/train")
    args.add_argument('-test_path', type=str, default='C:/Users/user/Desktop/test')
    args.add_argument('-log_path', type=str, default='C:/Users/user/Desktop/log')
    args.add_argument('-save_path', type=str, default='C:/Users/user/Desktop/save')
    args.add_argument('-iteration', type=int, default=1000)
    args.add_argument('-batch', type=int, default=16)
    config, _ = args.parse_known_args()

    train_data = Train_generator(config.train_path)
    test_data = Test_generator(config.test_path)
    model = Alexnet()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model.train(sess, train_data, config.log_path, config.iteration, config.batch)
    prob = model.test(sess, test_data)
    for idx, gender in enumerate(prob):
        print("%g번째 사진이 남성일 확률 : %g, 여성일 확률 : %g" % (idx, gender[0], gender[1]))
    model.save(sess, config.save_path)
    model.restore(config.save_path)

