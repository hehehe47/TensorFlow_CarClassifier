import tensorflow as tf
import os
from PIL import Image
import operator
# import Train_data
import numpy as np



def pred(pixel, cwd, file_name, rgb, kernel_size, train_batch, step,prob):
    labels = 5
    # cwd = 'testpic/'
    pixels = pixel
    cwd += '/'
    # print(cwd)
    # file_name = file_name
    pre_file_name = os.getcwd() + '/tfrecords/predict_set.tfrecords'
    # print(pre_file_name)
    img_dic = {}
    classes = {'suv': 0, 'car': 2, 'mb': 3, 'mini': 4, 'truck': 1}  # import images,, 'mb', 'mini', 'truck'

    pred_prob = 1.
    file_name = file_name + '_' + str(train_batch) + '_' + str(pixels) + '_' + str(step) + '_' + str(prob) + '_' + str(kernel_size)
    print(file_name)

    def create_test_set(file, pixels):
        count = 0
        if os.path.exists(cwd):
            writer = tf.python_io.TFRecordWriter(file)
            for img_name in os.listdir(cwd):
                # print(img_name)
                img_path = os.path.join(cwd, img_name)
                img = Image.open(img_path)
                img = img.resize((pixels, pixels))
                img_raw = img.tobytes()
                index = 0
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
                count += 1
                print(img_path + ' complete!')
            writer.close()
        return count

    def read_and_decode(file, pixels):
        # 创建文件队列,不限读取的数量
        filename_queue = tf.train.string_input_producer([file])
        # create a reader from file queue
        reader = tf.TFRecordReader()
        # reader从文件队列中读入一个序列化的样本
        _, serialized_example = reader.read(filename_queue)
        # get feature from serialized example
        # 解析符号化的样本
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        label = features['label']
        img = features['img_raw']
        # print(img)
        img = tf.decode_raw(img, tf.uint8)
        # print(img)
        img = tf.reshape(img, [pixels * pixels * rgb])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(label, tf.int32)
        return img, label

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
        h_fc1_drop = tf.nn.dropout(h_fc1, pred_prob)
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

    count = create_test_set(pre_file_name, pixels)
    print('1')
    img, label = read_and_decode(pre_file_name, pixels)
    print('2')
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size=count, capacity=2000,
        min_after_dequeue=0)

    # prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess,os.getcwd() + "/checkpoints/checkpoins_" + file_name  + '.ckpt')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image, label = sess.run([img_batch, label_batch])
    train_label3 = tf.one_hot(indices=label,
                              depth=labels,
                              on_value=1.0,
                              off_value=0.0)
    label1 = sess.run(train_label3)
    try:
        os.mkdir(os.getcwd() + '/pre_logs/')
    except:
        pass
    f = open(os.getcwd() + '/pre_logs/log_' + file_name  + '.txt', 'w')
    predi = sess.run(y_conv, feed_dict={x: image, keep_prob: 1.})
    # print(predi)
    # print(pred)
    j = 0
    car_list = os.listdir(cwd)
    for i in predi:
        # print(i)
        for index, value in enumerate(predi[j]):
            print(list(classes.keys())[list(classes.values()).index(index)], 'is ', value * 100, '%', end='    ')
        print(' ')
        print('picture', car_list[j], ' is predicted as a :', end=' ')
        max_index, max_value = max(enumerate(predi[j]), key=operator.itemgetter(1))
        print(list(classes.keys())[list(classes.values()).index(max_index)])
        f.write('picture ' + str(car_list[j]) + ' is predicted as a : ' + str(
            list(classes.keys())[list(classes.values()).index(max_index)]) + '\n')
        j += 1

    coord.request_stop()
    coord.join(threads=threads)
    # print('1')
    sess.close()
    # print(file_name)
    txt_file = 'D:\Python\CNN_4_CAR_TYPE13\pre_logs\log_' + file_name + '.txt'
    # print(txt_file)
    os.popen(txt_file)


# Train_data.Td(100, 8, 100, 'test', 0.5, 3, 3)
# pred(28, 'D:/Python/CNN_4_CAR_TYPE13/testpic', 'test', 3, 3, 200, 1000,0.5)
# def pred(pixel, cwd, file_name, rgb, kernel_size, train_batch, step):
