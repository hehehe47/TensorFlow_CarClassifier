import tensorflow as tf
import datetime
import os
import numpy as np
import Create_sets as Cs


def Td(train_batch, px, step, file_name, prob, rgb, kernel_size):
    cwd = 'garage'
    labels = 5
    train_batch_size = train_batch
    test_batch_size = 172

    # prob = 0.5
    pixels = px

    # file_name = 'test3'

    # 初始化单个卷积核上的参数
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    # 初始化单个卷积核上的偏置值
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    # 输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
    # padding表示是否需要补齐边缘像素使输出图像大小不变
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 对x进行最大池化操作，ksize进行池化的范围，
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, pixels * pixels * 3], name='x_input')

        y_ = tf.placeholder(tf.float32, [None, labels], name='y_input')

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('Layer1'):
        with tf.name_scope('Weitht'):
            W_conv1 = weight_variable([kernel_size, kernel_size, rgb, 32], name='W_conv1')
            # tf.add_to_collection('vars',W_conv1)
            # print("W_conv1: " ,W_conv1)
            tf.summary.histogram('Layer1/weight', W_conv1)
        with tf.name_scope('Biases'):
            b_conv1 = bias_variable([32], name="b_conv1")
            # tf.add_to_collection('vars', b_conv1)
            # print("b_conv1: ", b_conv1)
            tf.summary.histogram('Layer1/biase', b_conv1)
        x_image = tf.reshape(x, [-1, pixels, pixels, rgb])
        # print(x_image.shape)
        # 进行卷积操作，并添加relu激活函数
    with tf.name_scope('Conv_for_Layer1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # print("h_conv1: ", h_conv1)
        tf.summary.histogram('Layer1/conv1', h_conv1)
        # 进行最大池化
    with tf.name_scope('Pooling_for_Layer1'):
        h_pool1 = max_pool_2x2(h_conv1)
        # print("h_pool1: ", h_pool1)
        tf.summary.histogram('Layer1/pool1', h_pool1)

    # 同理第二层卷积层
    with tf.name_scope('Layer2'):
        with tf.name_scope('Weithts'):
            W_conv2 = weight_variable([kernel_size, kernel_size, 32, 64], name='W_conv2')
            # tf.add_to_collection('vars', W_conv2)
            # print("W_conv2: ", W_conv2)
            tf.summary.histogram('Layer2/weight', W_conv2)
        with tf.name_scope('Biases'):
            b_conv2 = bias_variable([64], name='b_conv2')
            # tf.add_to_collection('vars', b_conv2)
            # print("b_conv2: ", b_conv2)
            tf.summary.histogram('Layer2/biase', b_conv2)
    with tf.name_scope('Conv_for_Layer2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # print("h_conv2: ", h_conv2)
        tf.summary.histogram('Layer2/conv1', h_conv2)
    with tf.name_scope('Pooling_for_Layer2'):
        h_pool2 = max_pool_2x2(h_conv2)
        # print("h_pool2: ", h_pool2)
        tf.summary.histogram('Layer2/pool1', h_pool2)

    # 全连接层
    # 权值参数
    pixel = pixels / 4
    pixelsa = int(pixel)
    # print(pixelsa)
    W_fc1 = weight_variable([pixelsa * pixelsa * 64, 1024], name='W_fc1')
    # tf.add_to_collection('vars', W_fc1)
    # print("W_fc1: ", W_fc1)
    # 偏置值
    b_fc1 = bias_variable([1024], name='b_fc1')
    # tf.add_to_collection('vars', b_fc1)
    # print("b_fc1: ", b_fc1)
    # 将卷积的产出展开
    h_pool2_flat = tf.reshape(h_pool2, [-1, pixelsa * pixelsa * 64])
    # print("h_pool2_flat: ", h_pool2_flat)
    # 神经网络计算，并添加relu激活函数
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # print("h_fc1: ", h_fc1)

    # Dropout层，可控制是否有一定几率的神经元失效，防止过拟合，训练时使用，测试时不使用
    # keep_prob = tf.placeholder(tf.float32)
    # Dropout计算
    with tf.name_scope('Drop_out'):
        h_fc1_drop = tf.nn.dropout(h_fc1, prob)
        # print("h_fc1_drop: ", h_fc1_drop)
    # 输出层，使用softmax进行多分类
    W_fc2 = weight_variable([1024, labels], name='W_fc2')
    # tf.add_to_collection('vars', W_fc2)
    # print("W_fc2: ", W_fc2)
    b_fc2 = bias_variable([labels], name='b_fc2')
    # tf.add_to_collection('vars', b_fc2)
    # print("b_fc2: ", b_fc2)
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # print("y_conv: ", y_conv)

    # 读取二进制数据
    def read_and_decode(filename, pixels):
        filename_queue = tf.train.string_input_producer([filename], shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [pixels * pixels * rgb])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)

        return img, label

    img, label = read_and_decode(os.getcwd() + "/tfrecords/train.tfrecords", pixels)
    img1, label1 = read_and_decode(os.getcwd() + "/tfrecords/test.tfrecords", pixels)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size=train_batch_size, capacity=2000,  # 1958 + 122 + 234 + 176 + 2225
        min_after_dequeue=100)
    img_batch1, label_batch1 = tf.train.shuffle_batch(
        [img1, label1],
        batch_size=test_batch_size, capacity=2000,
        min_after_dequeue=100)
    # 初始化所有的op



    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), name='loss')
        # print("cross_entropy: ", cross_entropy)
        tf.summary.scalar('Loss', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # print("train_step: ", train_step)
    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # print("correct_prediction: ", correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("accuracy: ", accuracy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            os.mkdir(os.getcwd() + '/logs_graph/'+ str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size)+ '/train/')
            os.mkdir(os.getcwd() + '/logs_graph/' + str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size) + '/test/')
        except:
            pass
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            os.getcwd() + '/logs_graph/'+ str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size)+ '/train/', sess.graph)
        merged1 = tf.summary.merge_all()
        writer1 = tf.summary.FileWriter(
            os.getcwd() + '/logs_graph/' + str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size) + '/test/', sess.graph)
        sess.run(init)
        # 启动队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        begin = datetime.datetime.now()
        try:
            os.mkdir(os.getcwd() + '/logs/')
        except:
            pass
        f = open(os.getcwd() + '/logs/log_' +file_name + '_' + str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size) + '.txt', 'w')
        f.write('import parameters of this file:' + '\n')
        f.write('train_batch_size: ' + str(train_batch_size) + '\n')
        f.write('test_batch_size: ' + str(test_batch_size) + '\n')
        f.write('pixels: ' + str(pixels) + '\n')
        f.write('file_name: ' + str(file_name) + '\n')
        f.write('step: ' + str(step) + '\n')
        f.write('drop prob: ' + str(prob) + '\n')
        f.write('kernel_size: ' + str(kernel_size) + '\n')
        f.write('rgb: ' + str(rgb) + '\n')
        for i in range(step + 1):
            test_img, l = sess.run([img_batch, label_batch])
            test_img1, l1 = sess.run([img_batch1, label_batch1])

            train_label3 = tf.one_hot(indices=l1,
                                      depth=labels,
                                      on_value=1.0,
                                      off_value=0.0)
            label1 = sess.run(train_label3)
            train_label2 = tf.one_hot(indices=l,
                                      depth=labels,
                                      on_value=1.0,
                                      off_value=0.0)

            label = sess.run(train_label2)

            if (i) % (step * 0.02) == 0:
                result_train = sess.run(merged, feed_dict={x: test_img, y_: label, keep_prob: 1.0})
                result_test = sess.run(merged1, feed_dict={x: test_img1, y_: label1, keep_prob: 1.0})
                writer.add_summary(result_train, i)
                writer1.add_summary(result_test, i)
                train_accuracy = accuracy.eval(feed_dict={x: test_img, y_: label, keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={x: test_img1, y_: label1, keep_prob: 1.0})
                print(train_accuracy)
                print('test accuracy: ', test_accuracy)

                end = datetime.datetime.now()
                end = end - begin
                f.write('the train ' + str(i) + '\'s accuracy:' + str(train_accuracy) + '\n')
                f.write('the ' + str(i) + '\'s accuracy:' + str(test_accuracy) + '\n')
                f.write('the ' + str(i) + '\'s time:' + str(end) + '\n')
            sess.run(train_step, feed_dict={x: test_img, y_: label, keep_prob: prob})
            # sess.run(train_step, feed_dict={x: test_img1, y_: label1, keep_prob: 0.7})
        test_img1, l1 = sess.run([img_batch1, label_batch1])

        train_label3 = tf.one_hot(indices=l1,
                                  depth=labels,
                                  on_value=1.0,
                                  off_value=0.0)
        label1 = sess.run(train_label3)
        total_test_accuracy = accuracy.eval(feed_dict={x: test_img1, y_: label1, keep_prob: 1.0})
        f.write('total accuracy: ' + str(total_test_accuracy) + '\n')
        end = datetime.datetime.now()
        end = end - begin
        f.write('total: ' + str(end))
        f.close()
        try:
            os.mkdir(os.getcwd() + "/checkpoints/")
        except:
            pass
        # cpkt_name =
        saver.save(sess, os.getcwd() + "/checkpoints/checkpoins_" + file_name + '_' + str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size) + '.ckpt')

        coord.request_stop()
        # coord.request_stop()
        coord.join(threads=threads)

    print('train over!')

    # sess.run(init)
    sess.close()
    txt_file = 'D:\Python\CNN_4_CAR_TYPE13\logs\log_' +file_name + '_' + str(train_batch) + '_' + str(
            pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size) + '.txt'
    os.system(txt_file)
    # os.system('shutdown -s')

#
# pixels = 28
# Cs.Cs(pixels,1,False,0.4,0,3)
# Td(200, pixels, 1000, 'test', 1., 3, 3)
