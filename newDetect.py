import time

import cv2
import numpy as np
from ultralytics import YOLO


def get_fixed_windows(wind_size, overlap_size, image: np.ndarray):
    '''
    This function can generate overlapped windows given various image size
    params:
        image_size (w, h): the image width and height
        wind_size (w, h): the window width and height
        overlap (overlap_w, overlap_h): the overlap size contains x-axis and y-axis

    return:
        rects [(xmin, ymin, xmax, ymax)]: the windows in a list of rectangles
    '''
    rects = np.zeros((100, wind_size[1], wind_size[0], 3), dtype=np.uint8)
    image_size = (image.shape[1], image.shape[0])
    assert overlap_size[0] < wind_size[0]
    assert overlap_size[1] < wind_size[1]

    im_w = wind_size[0] if image_size[0] < wind_size[0] else image_size[0]
    im_h = wind_size[1] if image_size[1] < wind_size[1] else image_size[1]

    stride_w = wind_size[0] - overlap_size[0]
    stride_h = wind_size[1] - overlap_size[1]

    k = 0

    for j in range(wind_size[1] - 1, im_h + stride_h, stride_h):
        for i in range(wind_size[0] - 1, im_w + stride_w, stride_w):
            right, down = i + 1, j + 1
            right = right if right < im_w else im_w
            down = down if down < im_h else im_h

            left = right - wind_size[0]
            up = down - wind_size[1]

            rects[k] = image[up:down, left:right]

            k += 1
    return rects, k


def preProcessImage(image: np.ndarray, kernelSize):
    wind_size = kernelSize
    overlap_size = (300, 200)
    rets, n = get_fixed_windows(wind_size, overlap_size, image)
    return rets, n


def detect(image: np.ndarray, model, kernelSize):
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 648, 648)
    start = time.time()
    rets, n = preProcessImage(image, kernelSize)
    end = time.time()
    print("Preprocess time: ", end - start)
    for i in range(n):
        start = time.time()
        result = model.predict(rets[i])
        end = time.time()
        print("FPS: ", 1 / (end - start))
        for j in result:
            for c in j.boxes:
                b = c.xyxy[0]
                cls = c.cls
                label = model.names[int(cls)]
                cv2.putText(rets[i], label, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(rets[i], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.imshow("result", rets[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread('test.png')
    model = YOLO("yolov8n.pt")
    kernelSize = (648, 648)
    rets, n = preProcessImage(img, kernelSize)
    detect(img, model, kernelSize)
