import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow import keras
from resnet import resnet38
from dataset_production import get_files,get_tensor

tf.random.set_seed(2345)
# 训练和测试图片的路径
train_dir = 'F:\\Resnet\\data'
test_dir = 'D:\\pic\\test'

def proprecess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def train(db_train,db_test):
    model = resnet38()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:

                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in db_test:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

    model.save('path_to_saved_model', save_format='tf')
    # new_model = keras.models.load_model('path_to_saved_model')


if __name__ == '__main__':
    image_list, label_list = get_files(train_dir)
    # 测试图片与标签
    test_image_list, test_label_list = get_files(test_dir)
    # for i in range(len(image_list)):
    # print('图片路径 [{}] : 类型 [{}]'.format(image_list[i], label_list[i]))
    x_train, y_train = get_tensor(image_list, label_list)
    x_test, y_test = get_tensor(test_image_list, test_label_list)
    # print('image_list:{}, label_list{}'.format(image_list, label_list))
    print('----------------------Successful----------------------------')
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # # shuffle:打乱数据,map:数据预处理，batch:一次取喂入10样本训练
    db_train = db_train.shuffle(1000).map(proprecess).batch(10)
    # 载入训练数据集
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # # shuffle:打乱数据,map:数据预处理，batch:一次取喂入10样本训练
    db_test = db_test.shuffle(1000).map(proprecess).batch(10)
    # 生成一个迭代器输出查看其形状
    sample_train = next(iter(db_train))
    sample_test = next(iter(db_test))
    print('-------------data_init-----------------')
    train(db_train,db_test )
