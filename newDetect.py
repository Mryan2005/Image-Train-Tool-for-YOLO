import time

import cv2
import numpy as np
from ultralytics import YOLO


def getFixedWindows(kernelSize, overlapSize, image: np.ndarray):
    '''
    此函数可在给定不同图像尺寸的情况下生成重叠窗口
    参数:
        kernelSize (w, h): 卷积核的宽度和高度
        overlap (overlapW, overlapH): 重叠尺寸

    返回值:
        rects: a list of windows
        k: the number of windows
    '''
    rects = np.zeros((100, kernelSize[1], kernelSize[0], 3), dtype=np.uint8)
    imageSize = (image.shape[1], image.shape[0])
    assert overlapSize[0] < kernelSize[0]
    assert overlapSize[1] < kernelSize[1]

    imgW = kernelSize[0] if imageSize[0] < kernelSize[0] else imageSize[0]
    imgH = kernelSize[1] if imageSize[1] < kernelSize[1] else imageSize[1]

    strideW = kernelSize[0] - overlapSize[0]
    strideH = kernelSize[1] - overlapSize[1]

    k = 0

    for j in range(kernelSize[1] - 1, imgH + strideH, strideH):
        for i in range(kernelSize[0] - 1, imgW + strideW, strideW):
            right, down = i + 1, j + 1
            right = right if right < imgW else imgW
            down = down if down < imgH else imgH

            left = right - kernelSize[0]
            up = down - kernelSize[1]

            rects[k] = image[up:down, left:right]

            k += 1
    return rects, k


def preProcessImage(image: np.ndarray, kernelSize):
    overlapSize = (300, 200)
    rets, n = getFixedWindows(kernelSize, overlapSize, image)
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
        cv2.waitKey(3)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread('test.png')
    model = YOLO("yolov8n.onnx", task="detect")
    kernelSize = (640, 640)
    rets, n = preProcessImage(img, kernelSize)
    detect(img, model, kernelSize)
