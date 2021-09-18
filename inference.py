import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
from alexnet_finetune import AlexnetFinetune
from PIL import Image
import numpy as np
from matplotlib import cm
import time





def main():
    #Load model
    print("inference")
    path_img = "spray.ppm"
    nn = AlexnetFinetune()
    nn.model.load_state_dict(torch.load("state_dict_model_1.pt", map_location=torch.device('cpu')))
    nn.model.eval()

    #image = Image.open(path_img)
    #x = nn.predict_image(image)
    #print(x)

    x1 = 384
    y1 = 104
    x2 = 896
    y2 = 616
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        im = frame[y1:y2, x1:x2]

        cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        im.show()
        x = nn.predict_image(im)
        print(x)
        response = x

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, response, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), lineType=cv2.LINE_AA)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")

    print("inference")


if __name__ == "__main__":
    main()
