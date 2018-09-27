import tensorflow as tf

channel = [64, 128, 256, 512]
def bottleneck(input, channel_out, stride, index, scope):
    channel_in = input.get_shape()[-1]
    with tf.variable_scope(scope):
        first_layer = tf.layers.conv2d(input, filters=channel[index], kernel_size=1, strides=stride,
                                       padding='SAME', activation=tf.nn.relu, name='conv1_1x1')
        print('first', first_layer.shape)
        second_layer = tf.layers.conv2d(first_layer, filters=channel[index], kernel_size=3, strides=1,
                                        padding='SAME', activation=tf.nn.relu, name='conv2_3x3')
        print('second', second_layer.shape)
        third_layer = tf.layers.conv2d(second_layer, filters=channel_out, kernel_size=1, strides=1,
                                       padding='SAME', name='conv3_1x1')
        print('third', third_layer.shape)
        if channel_in != channel_out:
            # projection (option B)
            shortcut = tf.layers.conv2d(input, filters=channel_out, kernel_size=1,
                                        strides=stride, name='projection')
        else:
            shortcut = input   # identify
        output = tf.nn.relu(shortcut + third_layer)
        return output


def residual_block(input, channel_out, stride, n_bottleneck, index, down_sampling, scope):
    with tf.variable_scope(scope):
        if down_sampling:
            out = bottleneck(input, channel_out, stride=2, index=index, scope='bottleneck_1')
        else:
            out = bottleneck(input, channel_out, stride, index, scope='bottleneck_1')
        for i in range(1, n_bottleneck):
            out = bottleneck(out, channel_out, stride, index, scope='bottleneck_%i' % (i+1))
        return out

def ResNet_50(images):
    with tf.variable_scope('Layer_50'):
        # conv_1
        print('conv_1')
        conv_1 = tf.layers.conv2d(images, filters=64, kernel_size=7, strides=2, padding='SAME',
                                  activation=tf.nn.relu, name='conv1')
        print('conv_1', conv_1.shape)
        # conv2_x
        print('conv_2')
        max_pooling = tf.layers.max_pooling2d(conv_1, pool_size=3, strides=2,
                                              padding='SAME', name='max_pooling')
        conv_2 = residual_block(max_pooling, 256, stride=1, n_bottleneck=3, index=0,
                                down_sampling=False, scope='conv2')
        print('conv_2', conv_2.shape)
        # conv3_x
        print('conv_3')
        conv_3 = residual_block(conv_2, 512, stride=1, n_bottleneck=4, index=1,
                                down_sampling=True, scope='conv_3')
        print('conv_3:', conv_3.shape)
        # conv4_x
        print('conv_4')
        conv_4 = residual_block(conv_3, 1024, stride=1, n_bottleneck=6, index=2,
                                down_sampling=True, scope='conv4')
        print('conv_4', conv_4.shape)
        # conv5_x
        print('conv_5')
        conv_5 = residual_block(conv_4, 2048, stride=1, n_bottleneck=3, index=3,
                                down_sampling=True, scope='conv5')
        print('conv_5:', conv_5.shape)
        average_pooling = tf.layers.average_pooling2d(conv_5, pool_size=7, strides=1, name='avg_pooling')
        full_connection = tf.layers.flatten(average_pooling)
        print(full_connection.shape)
        logits = tf.nn.softmax(tf.layers.dense(full_connection, 1000, name='full_connection'))
        return logits

batch_size = 128
def main(argv=None):
    images = tf.zeros(shape=[batch_size, 224, 224, 3], dtype=tf.float32)
    ResNet_50(images)

if __name__ == '__main__':
    tf.app.run()


