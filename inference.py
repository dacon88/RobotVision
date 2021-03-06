"""
inference application opens the web-cam and predicts the obj based on the Alexnet NN finetuned
"""

import torch
from alexnet_finetune import AlexnetFinetune
from PIL import Image
import cv2
import time


def main():
    # TODO: handle input with argparse

    print("Load model")
    img_size = 160
    nn = AlexnetFinetune(img_size, "infer")
    nn.model.load_state_dict(torch.load("state_dict_model.pt", map_location=torch.device('cpu')))
    nn.model.eval()

    nn.get_classes_names_from_csv("classes_names.csv")

    # predict single img as smoke test
    # single_img = "lemon.ppm"
    # single_img = "test_img/pepper.ppm"
    # im = Image.open(single_img)
    # prediction = nn.predict_image(im)
    # print(prediction)

    # get img data from webcam
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print("inference")

    # Predict on the fly images
    while rval:

        # start_time = time.time()

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
        # im_rgb.save("PIL_input_rgb.jpg")

        x, predictions = nn.predict_image(im_rgb)
        print(predictions)
        # print(x)
        response = x

        # Render to screen
        cv2.putText(frame, response, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), lineType=cv2.LINE_AA)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        # print("--- %s seconds ---" % (time.time() - start_time))

        key = cv2.waitKey(150)  # 20 ms frame
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    main()
