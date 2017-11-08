from PIL import Image
import os
import tensorflow as tf
import shutil

from scipy.misc import imread, imresize


def Cs(pixel, create_flag, show_img_or_not, test_ratio, separate_flag, rgb):
    classes = {'suv': 0, 'car': 2, 'mb': 3, 'mini': 4, 'truck': 1}  # import images,, 'mb', 'mini', 'truck'

    try:
        os.mkdir(os.getcwd() + '/tfrecords/')
    except:
        pass

    def create(record, cwd, pixels):
        num = 0
        writer = tf.python_io.TFRecordWriter(record)
        for names in classes:
            class_path = cwd + "/" + names + "/"
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)  # .convert('L')  # change color of image
                img = img.resize((pixels, pixels))
                img_raw = img.tobytes()  # 将图片转化为原生bytes
                # print(img_raw)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[classes[names]])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
                print(img_path + ' complete!')
                num += 1
        writer.close()
        print(num)
        return num

    def create_tfrecord(cwd, record, pixels):
        if not create_flag:
            if not os.path.isfile(record):
                num = create(record, cwd, pixels)
                # print('1')
        else:
            if not os.path.isfile(record):
                num = create(record, cwd, pixels)
                # print('2')
            else:
                os.remove(record)
                num = create(record, cwd, pixels)
                # print('3')
        return num

    def show_img(cwd, record, num, pixels):
        filename_queue = tf.train.string_input_producer([record])  # 读入流中
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })  # 取出包含image和label的feature对象
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        try:
            image = tf.reshape(image, [pixels, pixels, rgb])
        except Exception as e:
            print(e)
        label = tf.cast(features['label'], tf.int32)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(num):
                example, l = sess.run([image, label])  # 在会话中取出image和label
                img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
                try:
                    os.mkdir(cwd)
                except:
                    pass
                img.save(cwd + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
                # print(example, l)
                print(i, ' complete!')
        coord.request_stop()
        coord.join(threads)
        # os.system('D:\Python\CNN_4_CAR_TYPE13/'+cwd)

    def count_image(garage):
        cwd = os.getcwd()
        cwd += garage
        total_num = car_num = mini_num = truck_num = suv_num = mb_num = 0
        for i in os.listdir(cwd):
            cwd_car = cwd + '/' + i
            for j in os.listdir(cwd_car):
                total_num += 1
                if i == 'mini':
                    mini_num += 1
                elif i == 'car':
                    car_num += 1
                elif i == 'truck':
                    truck_num += 1
                elif i == 'suv':
                    suv_num += 1
                elif i == 'mb':
                    mb_num += 1
        return total_num, car_num, mini_num, truck_num, suv_num, mb_num

    def separate_image(garage, ratio):
        cwd = os.getcwd()
        cwd += garage
        for i in os.listdir(cwd):
            cwd_car = cwd + '/' + i
            count = 0
            if i == 'mini':
                num = mini_num
            elif i == 'car':
                num = car_num
            elif i == 'truck':
                num = truck_num
            elif i == 'suv':
                num = suv_num
            elif i == 'mb':
                num = mb_num
            else:
                num = 0
            for j in os.listdir(cwd_car):
                cwd_img = cwd_car + '/' + j
                if count <= int(num * (1 - ratio)):
                    try:
                        os.makedirs(train_cwd + i)
                        # print('1')
                    except:
                        pass
                    shutil.copyfile(cwd_img, train_cwd + i)
                else:
                    try:
                        os.makedirs(test_cwd + i)
                        # print('1')
                    except:
                        pass
                    shutil.copyfile(cwd_img, test_cwd + i)
                count += 1

    total_num, car_num, mini_num, truck_num, suv_num, mb_num = count_image('/garage')
    list = ['car', 'suv', 'mini', 'truck', 'mb']  #
    train_cwd = 'train_garage/'
    test_cwd = 'test_garage/'
    train_tfrecord = os.getcwd() + '/tfrecords/train.tfrecords'
    test_tfrecord = os.getcwd() + '/tfrecords/test.tfrecords'
    train_save_cwd = 'train_pic_4_show/'
    test_save_cwd = 'test_pic_4_show/'
    # show_img_or_not = False
    # test_ratio = 0.4
    labels = 5
    pixels = pixel
    # print(pixels)
    train_total_num = total_num * (1 - test_ratio)  # 4715
    test_total_num = total_num - train_total_num
    # print(total_num, car_num, mini_num, truck_num, suv_num, mb_num)
    if separate_flag == 1:
        separate_image('/garage', test_ratio)
    train_num = create_tfrecord(cwd=train_cwd, record=train_tfrecord, pixels=pixels)
    # print(train_num)
    test_num = create_tfrecord(cwd=test_cwd, record=test_tfrecord, pixels=pixels)

    if show_img_or_not:
        show_img(cwd=train_save_cwd, record=train_tfrecord, num=train_num, pixels=pixels)
        show_img(cwd=test_save_cwd, record=test_tfrecord, num=test_num, pixels=pixels)


