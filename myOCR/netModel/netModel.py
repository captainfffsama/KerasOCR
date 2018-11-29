from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU

class OCRNet:
    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        #注意这里若使用y_pred会导致sequence_length（0）<= X.
        #注意理解sequence_length(0)这个东西  这里是32  因为第41句中的img_w//poolsize（2）是sequence_length（0）
        y_pred = y_pred[:, :, :]
        #这里labels是实际的标签，y_pred是预测出来的标签，input_length是预测标签的长度，label_length是实际标签的长度
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build(img_w,img_h,depth,alphabetLength,absolute_max_string_len):
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512
        
        act='relu'
        if K.image_data_format()=='channels_first':
            input_shape=(depth,img_w,img_h)
        else:
            input_shape=(img_w,img_h,depth)

        input_data=Input(name='the_input',shape=input_shape,dtype='float32')
        inner =Conv2D(conv_filters,kernel_size,padding='same',activation=act,
                    kernel_initializer='he_normal',name='conv1')(input_data)
        
        inner =MaxPooling2D(pool_size=(pool_size,pool_size),name='max1')(inner)
        inner =Conv2D(conv_filters,kernel_size,padding='same',activation=act,
                    kernel_initializer='he_normal',name='conv2')(inner)
        
        inner =MaxPooling2D(pool_size=(pool_size,pool_size),name='max2')(inner)
        conv_to_rnn_dims = (img_w // (pool_size**2),(img_h // (pool_size**2 )) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)       

        gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
    
        gru_2 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
 
        # transforms RNN output to character activations:
        inner = Dense(alphabetLength, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(OCRNet.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
        return Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)