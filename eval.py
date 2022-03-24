import cv2
import numpy as np
import pyzed.sl as sl
import torch
import torchvision.transforms as transforms
from network import MyModel

if __name__ == '__main__':

    # Load DNN model
    model = torch.load(r"./model/model_best.pth", map_location=torch.device('cpu'))
    model.eval()

    # Image Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize(0.1307, 0.3081)
    ])

    # Creat camera
    zed = sl.Camera()

    # Camera Init
    zed_init_param = sl.InitParameters()
    zed_init_param.camera_resolution = sl.RESOLUTION.HD1080
    zed_init_param.camera_fps = 30

    # Open zed
    state = zed.open(zed_init_param)
    if state != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Set image windows
    cv2.namedWindow("zed", 0)
    cv2.resizeWindow("zed", 800, 450)
    cv2.namedWindow("ROI", 1)
    cv2.resizeWindow("ROI", 50, 50)

    # Capture frames of zed
    runtime_param = sl.RuntimeParameters()

    while True:
        img_zed = sl.Mat()
        if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS:
            # Geap an image
            zed.retrieve_image(img_zed, sl.VIEW.LEFT)   # Left Camera
            img = img_zed.get_data()
            (h, w) = img.shape[:2]
            img = img[..., 0:3]
            img = np.array(img, "uint8")

            # Set ROI
            (roi_h, roi_w) = (150, 150)
            roi_rect = ((w-roi_w)/2, (h-roi_h)/2, (w+roi_w)/2, (h+roi_h)/2)
            roi_rect = np.array(roi_rect, "int")
            img_roi = img[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2], :]
            img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)

            # Binarize ROI
            sucess, img_roi = cv2.threshold(img_roi, 135, 255, cv2.THRESH_BINARY_INV)

            # Number Identify
            roi_tensor = transform(img_roi).unsqueeze(0)

            num = model(roi_tensor)
            num = int(num.argmax(-1))

            # Mark in the Image
            cv2.rectangle(img, (roi_rect[0], roi_rect[1]),
                          (roi_rect[2], roi_rect[3]), (0, 255, 0), 2)
            cv2.putText(img, "The number is: {}".format(num), (roi_rect[0], roi_rect[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color=(0,255,0),
                        thickness=2)

            cv2.imshow("zed", img)
            cv2.imshow("ROI", img_roi)

            # Press "Esc" to quit the program
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                zed.close()
                break
