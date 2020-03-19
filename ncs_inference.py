from openvino.inference_engine import IENetwork, IEPlugin, IECore
import cv2
import numpy as np 

ie = IECore()

#ie.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'}, "MYRIAD")

net = IENetwork(model='./model/plate_model.xml', weights='./model/plate_model.bin')
plugin = IEPlugin(device="MYRIAD")
exec_net = plugin.load(network=net)


HEIGHT = 32
WIDTH = 32

def predict(img):
    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    img = cv2.resize(img, (WIDTH,HEIGHT))
    prediction = exec_net.infer({"conv2d_1_input": np.expand_dims(img, axis=0)})
    prediction = prediction['dense_4/Sigmoid'][0]
    
    return prediction

 
if __name__ == "__main__":

    for i in range(9):
    
        img = cv2.imread('./data/'+str(i)+'.png')
        cv2.imshow('img', img)
        cv2.waitKey(100)
        res = predict(img)
        print(res)
    
