import cv2


# TODO: handle commented lines
#cv2.namedWindow("preview")
#vc = cv2.VideoCapture(0)

#if vc.isOpened(): # try to get the first frame
#    rval, frame = vc.read()
#else:
#    rval = False

#while rval:
#    cv2.imshow("preview", frame)
#    rval, frame = vc.read()
#    key = cv2.waitKey(20)
#    if key == 27: # exit on ESC
#        break
#cv2.destroyWindow("preview")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
import PIL
from alexnet_finetune import AlexnetFinetune



def main():
    #Load model
    print("inference")

    #model = torchvision.models.alexnet(num_classes=7)

    #model.load_state_dict(torch.load("state_dict_model_1.pt", map_location=torch.device('cpu')))
    #model.eval()

    path_img = "phone.jpg"

    nn = AlexnetFinetune()
    nn.model.load_state_dict(torch.load("state_dict_model_1.pt", map_location=torch.device('cpu')))
    nn.model.eval()
    x = nn.predict_image(path_img)

    print("inference")


if __name__ == "__main__":
    main()
