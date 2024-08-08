import multiprocessing
import time

import cv2
import numba as nb
import numpy as np
from ultralytics import YOLO

cpuCount = multiprocessing.cpu_count()
nb.set_num_threads(cpuCount)


def detectSingleKernel(image: np.ndarray, model, rets, logIt, i):
    '''
    此函数用于检测单个窗口
    参数:
        image: 输入图像
        model: 模型
        kernelSize: 窗口大小
    '''
    start1 = time.time()
    result = model.predict(rets[i], conf=0.5)
    end1 = time.time()
    print("FPS: ", 1 / (end1 - start1))
    print("Detect time: ", end1 - start1)
    for j in result:
        for c in j.boxes:
            b = c.xyxy[0]
            cls = c.cls
            label = model.names[int(cls)]
            cv2.putText(rets[i], label, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(rets[i], (int(b[0]), int(b[1])), (int(b[2]), int(b[3]),), (0, 255, 0), 2)
            cv2.rectangle(image, (int(b[0] + logIt[i][0]), int(b[1] + logIt[i][1])), (int(b[2] + logIt[i][0]),
                                                                                      int(b[3] + logIt[i][1])),
                          (0, 255, 0), 2)


# @nb.jit(nopython=True, parallel=True, fastmath=True, cache=True)
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
    imageSize = (image.shape[1], image.shape[0])
    assert overlapSize[0] < kernelSize[0]
    assert overlapSize[1] < kernelSize[1]

    imgW = kernelSize[0] if imageSize[0] < kernelSize[0] else imageSize[0]
    imgH = kernelSize[1] if imageSize[1] < kernelSize[1] else imageSize[1]

    strideW = kernelSize[0] - overlapSize[0]
    strideH = kernelSize[1] - overlapSize[1]

    sum = 0

    for j in range(kernelSize[1] - 1, imgH + strideH, strideH):
        for i in range(kernelSize[0] - 1, imgW + strideW, strideW):
            sum += 1

    rects = np.zeros((sum, kernelSize[1], kernelSize[0], 3), dtype=np.uint8)
    logIt = np.zeros((sum, 2), dtype=np.uint64)

    k = 0

    for j in range(kernelSize[1] - 1, imgH + strideH, strideH):
        for i in range(kernelSize[0] - 1, imgW + strideW, strideW):
            right, down = i + 1, j + 1
            right = right if right < imgW else imgW
            down = down if down < imgH else imgH

            left = right - kernelSize[0]
            up = down - kernelSize[1]

            rects[k] = image[up:down, left:right]
            logIt[k][0] = left
            logIt[k][1] = up

            k += 1

    return rects, k, logIt


def preProcessImage(image: np.ndarray, kernelSize):
    overlapSize = (int(kernelSize[0] * 0.15), int(kernelSize[1] * 0.15))
    rets, n, logIt = getFixedWindows(kernelSize, overlapSize, image)
    return rets, n, logIt


def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters:
    boxes (np.ndarray): Array of bounding boxes with shape (N, 4), where N is the number of boxes.
                        Each box is represented as [x1, y1, x2, y2].
    scores (np.ndarray): Array of confidence scores with shape (N,).
    iou_threshold (float): IoU threshold for suppression.

    Returns:
    np.ndarray: Array of indices of the bounding boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Convert to float for precision
    boxes = boxes.astype(np.float32)

    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def detect(image: np.ndarray, model, kernelSize):
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 648, 648)
    start = time.time()
    rets, n, logIt = preProcessImage(image, kernelSize)
    end = time.time()
    print("Preprocess time: ", end - start)
    start = time.time()
    for i in range(n):
        detectSingleKernel(image, model, rets, logIt, i)
    for j in result:
        for c in j.boxes:
            b = c.xyxy[0]
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3]),), (0, 255, 0), 2)
    end = time.time()
    # image = cv2.resize(image, (640, 640))
    # result = model.predict(image)
    # for j in result:
    #     for c in j.boxes:
    #         b = c.xyxy[0]
    #         cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3]),), (0, 255, 0), 2)
    print("Total Detect time: ", end - start)
    print("FPS: ", 1 / (end - start))
    cv2.imshow("result", image)
    cv2.waitKey(30)


if __name__ == "__main__":
    img = cv2.imread('test.png')
    preProcessImage(img.copy(), (640, 640))
    model = YOLO("yolov5nu.onnx", task="detect")
    result = model.predict(img, conf=0.5)
    kernelSize = (640, 640)
    for i in range(10):
        detect(img.copy(), model, kernelSize)
    cv2.destroyAllWindows()
