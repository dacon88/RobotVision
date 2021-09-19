"""
inference application opens the webcam and predicts the obj based on the Alexnet NN finetuned
"""

import torch
from alexnet_finetune import AlexnetFinetune
from PIL import Image
import cv2


def main():

    print("Load model")
    nn = AlexnetFinetune()
    nn.model.load_state_dict(torch.load("state_dict_model.pt", map_location=torch.device('cpu')))
    nn.model.eval()

    nn.classes_names = nn.get_classes_names_from_csv("classes_names.csv")

    # get img data from webcam
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print("inference")

    # Predict on the fly images
    while rval:

        img_bgr = frame

        # process camera img
        # resize
        new_resize_dim = (160, 160)
        img_bgr = cv2.resize(img_bgr, new_resize_dim, interpolation=cv2.INTER_AREA)

        # write img to disk for debugging
        # cv2.imwrite("cv2.jpg", img_bgr)

        # Convert CV2 to PIL and from RGB to BRG. Note: could not manage to have cv2.cvtColor working
        im_pil = Image.fromarray(img_bgr)
        R, G, B = im_pil.split()
        im_rgb = Image.merge("RGB", (B, G, R))

        # write img to disk for debugging
        img_to_disk = im_rgb.save("PIL_input_rgb.jpg")

        x = nn.predict_image(im_rgb)
        # print(x)
        response = x

        # Render to screen
        cv2.putText(frame, response, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), lineType=cv2.LINE_AA)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    main()
