"""
inference application opens the webcam and predicts the obj based on the Alexnet NN finetuned
"""

import torch
from alexnet_finetune import AlexnetFinetune
from PIL import Image
import cv2


def main():

    #Load model
    nn = AlexnetFinetune()
    nn.model.load_state_dict(torch.load("state_dict_model.pt", map_location=torch.device('cpu')))
    nn.model.eval()

    # pixel coord for cropping the webcam img
    x1 = 384
    y1 = 104

    x2 = 896
    y2 = 616

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        #TODO: fix this variable horible thing
        img_bgr = frame
        #TODO: maybe crop on purpose
        #img_rgb = frame[y1:y2, x1:x2]

        #cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dim = (160, 160)
        img_bgr = cv2.resize(img_bgr, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("cv2.jpg", img_bgr)
        im_PIL = Image.fromarray(img_bgr)
        R, G, B = im_PIL.split()

        im_rgb = Image.merge("RGB", (B, G, R))
        # im.show()
        im1 = im_rgb.save("input_rgb.jpg")

        x = nn.predict_image(im_rgb)
        print(x)
        response = x

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
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
