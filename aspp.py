# coding=utf-8
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow as tf
from tensorflow.keras import backend as K

# ref:https://github.com/rishizek/tensorflow-deeplab-v3-plus/blob/master/deeplab_model.py
# https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
def aspp(x, depth=256, atrous_rates=[6, 12, 18]):

  conv_1x1 = Conv2D(depth, (1, 1), stride=1, scope="aspp_conv_1x1")(x)
  conv_1x1 = BatchNormalization(name='aspp_conv_1x1_BN', epsilon=1e-5)(conv_1x1)
  conv_1x1 = Activation('relu')(conv_1x1)
  conv_3x3_1 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[0], scope='aspp_conv_3x3_1')(x)
  conv_3x3_1 = BatchNormalization(name='aspp_conv_3x3_1_BN', epsilon=1e-5)(conv_3x3_1)
  conv_3x3_1 = Activation('relu')(conv_3x3_1)
  conv_3x3_2 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[1], scope='aspp_conv_3x3_2')(x)
  conv_3x3_2 = BatchNormalization(name='aspp_conv_3x3_2_BN', epsilon=1e-5)(conv_3x3_2)
  conv_3x3_2 = Activation('relu')(conv_3x3_2)
  conv_3x3_3 = Conv2D(depth, (3, 3), stride=1, rate=atrous_rates[2], scope='aspp_conv_3x3_3')(x)
  conv_3x3_3 = BatchNormalization(name='aspp_conv_3x3_3_BN', epsilon=1e-5)(conv_3x3_3)
  conv_3x3_3 = Activation('relu')(conv_3x3_3)
  
  b4 = GlobalAveragePooling2D()(x)
  # from (b_size, channels)->(b_size, 1, 1, channels)
  b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
  b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
  b4 = Conv2D(256, (1, 1), padding='same',
              use_bias=False, name='aspp_image_pooling')(b4)
  b4 = BatchNormalization(name='aspp_image_pooling_BN', epsilon=1e-5)(b4)
  b4 = Activation('relu')(b4)
  # upsample. have to use compat because of the option align_corners
  size_before = tf.keras.backend.int_shape(x)
  b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
  
  out = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, b4], axis=3, name='aspp_concat')
  out = Conv2D(depth, (1, 1), stride=1, scope='aspp_conv_1x1_concat')(out)
  out = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(out)
  out = Activation('relu')(
  return out