# coding=utf-8
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, Lambda, Concatenate
import tensorflow as tf
from tensorflow.keras import backend as K

# ref:https://github.com/rishizek/tensorflow-deeplab-v3-plus/blob/master/deeplab_model.py
# https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
def aspp(x, depth=256, atrous_rates=[6, 12, 18]):

  #print(x.shape)
  conv_1x1 = Conv2D(depth, (1, 1), padding='same', name="aspp_conv_1x1")(x)
  conv_1x1 = BatchNormalization(name='aspp_conv_1x1_BN', epsilon=1e-5)(conv_1x1)
  conv_1x1 = Activation('relu')(conv_1x1)
  conv_3x3_1 = Conv2D(depth, (3, 3), padding='same', dilation_rate=atrous_rates[0], name='aspp_conv_3x3_1')(x)
  conv_3x3_1 = BatchNormalization(name='aspp_conv_3x3_1_BN', epsilon=1e-5)(conv_3x3_1)
  conv_3x3_1 = Activation('relu')(conv_3x3_1)
  conv_3x3_2 = Conv2D(depth, (3, 3), padding='same', dilation_rate=atrous_rates[1], name='aspp_conv_3x3_2')(x)
  conv_3x3_2 = BatchNormalization(name='aspp_conv_3x3_2_BN', epsilon=1e-5)(conv_3x3_2)
  conv_3x3_2 = Activation('relu')(conv_3x3_2)
  conv_3x3_3 = Conv2D(depth, (3, 3), padding='same', dilation_rate=atrous_rates[2], name='aspp_conv_3x3_3')(x)
  conv_3x3_3 = BatchNormalization(name='aspp_conv_3x3_3_BN', epsilon=1e-5)(conv_3x3_3)
  conv_3x3_3 = Activation('relu')(conv_3x3_3)
  
  b4 = GlobalAveragePooling2D()(x)
  # from (b_size, channels)->(b_size, 1, 1, channels)
  b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
  b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
  b4 = Conv2D(depth, (1, 1), padding='same',
              use_bias=False, name='aspp_image_pooling')(b4)
  b4 = BatchNormalization(name='aspp_image_pooling_BN', epsilon=1e-5)(b4)
  b4 = Activation('relu')(b4)
  # upsample. have to use compat because of the option align_corners
  size_before = tf.shape(x)	# for None
  #size_before = x.shape	# for fixed size
  b4 = Lambda(lambda x: tf.image.resize_bilinear(x, size_before[1:3], align_corners=True))(b4)
  #print(b4.shape)
  
  out = Concatenate(axis=3, name='aspp_concat')([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, b4])
  out = Conv2D(depth, (1, 1), padding='same', name='aspp_conv_1x1_concat')(out)
  out = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(out)
  out = Activation('relu')(out)
  #print(out.shape)
  return out
