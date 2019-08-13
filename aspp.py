# coding=utf-8
from tensorflow.keras.layers import Conv2D
from tensorflow as tf

# ref:https://github.com/rishizek/tensorflow-deeplab-v3-plus/blob/master/deeplab_model.py
def aspp(x, depth=256, atrous_rates=[6, 12, 18]):

  inputs_size = tf.shape(x)[1:3]

  conv_1x1 = Conv2D(depth, (1, 1), stride=1, scope="aspp_conv_1x1")(x)
  conv_3x3_1 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[0], scope='aspp_conv_3x3_1')(x)
  conv_3x3_2 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[1], scope='aspp_conv_3x3_2')(x)
  conv_3x3_3 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[2], scope='aspp_conv_3x3_3')(x)
  
  # global average pooling
  image_level_features = tf.reduce_mean(x, (1, 2), name='global_average_pooling', keepdims=True)
  # 1x1 convolution with 256 filters( and batch normalization)
  image_level_features = Conv2D(depth, (1, 1), stride=1, scope='aspp_conv_1x1')(image_level_features)
  # bilinearly upsample features
  image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='aspp_upsample')
  
  out = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='aspp_concat')
  out = Conv2D(depth, (1, 1), stride=1, scope='aspp_conv_1x1_concat')(out)