from loadData import loadpic 
from netModel.netModel import OCRNet

from keras import backend as K
from keras.optimizers import SGD

from keras.callbacks import TensorBoard
from visual_callback import saveModel


def train(dataPath,testPath,batchSize,absolute_max_string_len,downsample_factor):
    #初始化参数
    model=OCRNet.build(128,64,3,28,absolute_max_string_len)
    
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.summary()
    saveModel_obj=saveModel()
    model.fit_generator(generator=loadpic(dataPath,batchSize,absolute_max_string_len,downsample_factor),
    steps_per_epoch=1000, epochs=10, 
    validation_data=loadpic(testPath,batchSize,absolute_max_string_len,downsample_factor), 
    validation_steps=50,
    callbacks=[TensorBoard(log_dir='./log')])
    model.save_weights('weights.h5')
trainFolder='D:/code/testAndExperiment/py/KerasOcr/pics'
testFolder=r'D:\code\testAndExperiment\py\KerasOcr\test'
train(trainFolder,testFolder,batchSize=32,absolute_max_string_len=12,downsample_factor=4)
