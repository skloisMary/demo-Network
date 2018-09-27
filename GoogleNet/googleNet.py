import tensorflow as tf
from tensorflow.contrib.layers import flatten


def Inception_Module(input, filter_1x1, fileter_3x3_reduce,
                     filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool_proj, scope):
    with tf.variable_scope(scope):
        conv_1x1 = tf.layers.conv2d(inputs=input, filters=filter_1x1, kernel_size=1, padding='SAME',
                                    activation=tf.nn.relu, name='conv_1x1')

        conv_3x3_reduce = tf.layers.conv2d(inputs=input, filters=fileter_3x3_reduce, kernel_size=1, padding='SAME',
                                           activation=tf.nn.relu, name='conv_3x3_reduce')
        conv_3x3 = tf.layers.conv2d(inputs=conv_3x3_reduce, filters=filter_3x3, kernel_size=3, padding='SAME',
                                    activation=tf.nn.relu, name='conv_3x3')

        conv_5x5_reduce = tf.layers.conv2d(inputs=input, filters=filter_5x5_reduce, kernel_size=1, padding='SAME',
                                           activation=tf.nn.relu, name='conv_5x5_reduce')
        conv_5x5 = tf.layers.conv2d(inputs=conv_5x5_reduce, filters=filter_5x5, kernel_size=5, padding='SAME',
                                    activation=tf.nn.relu, name='conv_5x5')

        maxpool = tf.layers.max_pooling2d(inputs=input, pool_size=3, strides=1, padding='SAME', name='max_pool')
        maxpool_proj = tf.layers.conv2d(inputs=maxpool, filters=filter_pool_proj, kernel_size=1, padding='SAME',
                                        activation=tf.nn.relu, name='maxpool_proj')
        inception_output = tf.concat((conv_1x1, conv_3x3, conv_5x5, maxpool_proj),
                                     axis=-1, name='inception_output')
        return inception_output


def GoogleNet(images):
    with tf.variable_scope('GoogleNet'):
        conv_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=7, strides=2,
                                 activation=tf.nn.relu, padding='SAME', name='conv_1')
        max_pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=3, strides=2,
                                             padding='SAME', name='max_pool_1')
        print('conv_1:', conv_1.shape, 'max_pool_1:', max_pool_1.shape)

        conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=192, kernel_size=3, strides=1, padding='SAME',
                                  activation=tf.nn.relu, name='conv_2')
        max_pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=3, strides=2, padding='SAME', name='max_pool_2')
        print('conv_2:', conv_2.shape, 'max_pooling_2', max_pool_2.shape)

        inception_3a = Inception_Module(max_pool_2, 64, 96, 128, 16, 32, 32, scope='inception_3a')
        inception_3b = Inception_Module(inception_3a, 128, 128, 192, 32, 96, 64, scope='inception_3b')
        max_pool_3 = tf.layers.max_pooling2d(inputs=inception_3b, pool_size=3, strides=2, padding='SAME', name='max_pool_3')
        print('inception_3a:', inception_3a.shape, 'inception_3b:', inception_3b.shape, 'max_pool_3:', max_pool_3.shape)

        inception_4a = Inception_Module(max_pool_3, 192, 96, 208, 16, 48, 64, scope='inception_4a')
        inception_4b = Inception_Module(inception_4a, 160, 112, 224, 24, 64, 64, scope='inception_4b')
        inception_4c = Inception_Module(inception_4b, 128, 128, 256, 24, 64, 64, scope='inception_4c')
        inception_4d = Inception_Module(inception_4c, 112, 144, 288, 32, 64, 64, scope='inception_4d')
        inception_4e = Inception_Module(inception_4d, 256, 160, 320, 32, 128, 128, scope='inception_4e')
        max_pool_4 = tf.layers.max_pooling2d(inputs=inception_4e, pool_size=3, strides=2, padding='SAME', name='max_pool_4')
        print('inception_4a', inception_4a.shape,
              'inception_4b', inception_4b.shape,
              'inception_4c', inception_4c.shape,
              'inception_4d', inception_4d.shape,
              'inception_4e', inception_4e.shape,
              'max_pool_4:', max_pool_4.shape)

        inception_5a = Inception_Module(max_pool_4, 256, 160, 320, 32, 128, 128, scope='inception_5a')
        inception_5b = Inception_Module(inception_5a, 384, 192, 384, 48, 128, 128, scope='inception_5b')
        avg_pool = tf.layers.average_pooling2d(inputs=inception_5b, pool_size=7, strides=1, padding='SAME', name='avg_pool')
        print('inception_5a:', inception_5a.shape, 'inception_5b', inception_5b.shape, 'avg_pool:', avg_pool.shape)

        drop_out = tf.layers.dropout(inputs=avg_pool, rate=0.4)
        flattern_layer = flatten(drop_out)
        print(flattern_layer.shape)
        linear = tf.layers.dense(inputs=flattern_layer, units=1000, name='linear')
        print(linear.shape)
        logits = tf.nn.softmax(linear)
        return logits

batch_size = 128
def main(argv=None):
    images = tf.zeros(shape=[batch_size, 224, 224, 3], dtype=tf.float32)
    GoogleNet(images)

if __name__ == '__main__':
    tf.app.run()