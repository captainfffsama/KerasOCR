from keras.models import Model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
from keras import backend as K
from allNumList import alphabet

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = labels_to_text(lables)
      results.append(text)
    return results

def test(modelPath,testPicTest):
    img=cv2.imread(testPicTest)
    img=cv2.resize(img,(128,64))
    img=img_to_array(img)
    img=np.array(img,dtype='float')/255.0
    img=np.expand_dims(img, axis=0)
    img=img.swapaxes(1,2)   
    
    model=load_model(modelPath,custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    net_out_value = model.predict(img)
    top_pred_texts = decode_predict_ctc(net_out_value)
    return top_pred_texts

result=test(r'D:\code\testAndExperiment\py\KerasOcr\weights.h5',r'D:\code\testAndExperiment\py\KerasOcr\test\avo.jpg') 
print(result)   
    
