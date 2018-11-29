from imutils import paths
import cv2
import random
from keras.preprocessing.image import img_to_array
import numpy as np
import allNumList 

def text_to_labels(text,alphabet):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

def loadpic(path,batchSize,absolute_max_string_len,downsample_factor):
    imgPaths=list(paths.list_images(path))
    random.seed(20)

    while 1:
        x_batch=[]#图片
        y_batch=np.ones([batchSize, absolute_max_string_len]) * -1#图片标签变成编码
        input_length = np.zeros([batchSize, 1])#图片下采样后进入GRN的尺寸
        label_length=np.zeros([batchSize, 1])#图片中字符的长度
        labels_batch=[]#图片的真是标签 字符串
        for i in range(batchSize):
                imgpath=imgPaths[random.randint(0,len(imgPaths)-1)]
                img=cv2.imread(imgpath,1)
                img=cv2.resize(img,(128,64))
                # cv2.imshow("ig",img)
                # cv2.waitKey(30)
                img=img_to_array(img)
                
                x_batch.append(img)

                input_length[i]=np.array(img).shape[1]//downsample_factor

                imgname=imgpath[imgpath.rindex('\\')+1:imgpath.rindex('.')]
                labels_batch.append(imgname)

                label_length[i]=len(imgname)

                y_batch[i,0:len(imgname)]=text_to_labels(imgname,allNumList.alphabet)
        x_batch = np.array(x_batch, dtype="float") / 255.0
        x_batch= x_batch.swapaxes(1,2)

        inputs = {'the_input': x_batch,
                  'the_labels': y_batch,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': labels_batch  # used for visualization only # 可视化使用
                  }
        outputs = {'ctc': np.zeros([batchSize])}  
        yield (inputs,outputs)



 
 
