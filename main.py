import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import split

class kernel:

    def __init__(self, kernel_size: tuple, beginPlace: tuple):
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size[0], kernel_size[1]), np.uint8)
        self.beginPlace = beginPlace
        self.detectResult = []

    def copy_to_kernel(self, image: np.ndarray) -> np.ndarray:
        # 获取卷积核的大小
        kernel_size = self.kernel.shape[0]

        # 获取图像的左上角区域
        image_part = image[self.beginPlace[0]:self.beginPlace[0]+kernel_size, self.beginPlace[1]:self.beginPlace[1]+kernel_size]

        # 将图像的部分复制到卷积核中
        self.kernel = image_part.copy()

        return self.kernel

def divideImageIntoNParts(image: np.ndarray, n: int):
    height, width, _ = image.shape
    stepLength = (height//n, width//n)
    parts: list = []
    for i in range(0, n, stepLength[0]):
        for j in range(0, n, stepLength[1]):
            ker = kernel(stepLength, (i, j))
            parts.append(ker)
    return parts

def detectIamge(image: np.ndarray, model: YOLO):
    parts = divideImageIntoNParts(image, 3)
    for part in parts:
        part.copy_to_kernel(image)
        result = model(part.kernel)
        part.detectResult = result
    return parts

def mergeParts(parts: list, image: np.ndarray):
    for part in parts:
        image[part.beginPlace[0]:part.beginPlace[0]+part.kernel_size[0], part.beginPlace[1]:part.beginPlace[1]+part.kernel_size[1]] = part.kernel
    return image

def drawDetectResult(image: np.ndarray, parts: list):
    for part in parts:
        for result in part.detectResult:
            for i in result.boxes:
                xyxy = i.xyxy[0]
                cls = i.cls
                label = result.names[int(cls)]
                cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), 255, 2)
    return image


class imgProcessStep:
    def __init__(self, image: np.ndarray, selectRetangle: tuple):
        self.image = image
        self.selectRetangle = selectRetangle


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item: np.ndarray):
        self.stack.append(item.copy())

    def pop(self) -> np.ndarray:
        if len(self.stack) == 0:
            return np.array([])
        data = self.stack.pop(-1)
        return data.copy()

    def top(self) -> np.ndarray:
        if len(self.stack) == 0:
            return np.array([])
        data = self.stack[-1]
        return data.copy()

    def clear(self):
        self.stack = []

    def length(self) -> int:
        return len(self.stack)


class Tool:
    def __init__(self, imageSource: list):
        self.workStack = Stack()
        self.workStackTemp = Stack()
        self.imageSource = imageSource
        self.selectRetangle = []
        self.saveRetangle = []
        self.windowName = "Tool"
        self.isSelecting = False
        self.count = 0

    def MouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.isSelecting:
            self.selectRetangle.append((x, y))
            self.isSelecting = True
        elif event == cv2.EVENT_LBUTTONDOWN and self.isSelecting:
            self.selectRetangle.append((x, y))
            self.isSelecting = False
            self.workStackTemp.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selectRetangle = []
            self.isSelecting = False

    def Run(self):
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName, self.MouseEvent)
        for image in self.imageSource:
            self.selectRetangle = []
            image_cv2: np.ndarray = cv2.imread(image)
            image_cv2 = cv2.resize(image_cv2, (int(800), int(800)))
            image_cv2_original = image_cv2.copy()
            self.workStack.push(image_cv2_original)
            self.count += 1
            if image_cv2 is None:
                print("Cannot read image: ", image)
                continue
            while True:
                if len(self.selectRetangle) == 2:
                    image_cv2 = cv2.rectangle(image_cv2, self.selectRetangle[0], self.selectRetangle[1], (0, 255, 0),
                                              2)  # type: ignore
                    a = imgProcessStep(image_cv2.copy(), self.selectRetangle)
                    self.saveRetangle.append((self.selectRetangle))
                    ima = image_cv2.copy()
                    self.workStack.push(ima)
                    self.selectRetangle = []
                cv2.imshow(self.windowName, image_cv2)  # type: ignore
                a = cv2.waitKey(1)
                if a == ord('Y') or a == ord('y'):
                    if self.workStackTemp.length() > 0:
                        self.workStack.push(self.workStackTemp.pop())
                        image_cv2 = self.workStack.top()
                    else:
                        print("Cannot redo.")
                elif a == ord('P') or a == ord('p'):
                    cv2.imwrite("datasets/original/images/" + str(self.count) + ".jpeg",
                                image_cv2_original)
                    file = open("datasets/original/labels/" + str(self.count) + ".txt", "w")
                    file.write('0 ')
                    for i in range(len(self.saveRetangle)):
                        file.write(str(self.saveRetangle[i][0][0] / 800) + ' ' + str(
                            self.saveRetangle[i][0][1] / 800) + ' ' + str(
                            self.saveRetangle[i][1][0] / 800) + ' ' + str(
                            self.saveRetangle[i][1][1] / 800) + '\n')
                    file.close()
                    print("Save image: ", "image/" + str(self.count) + ".jpeg")
                    for i in range(len(self.saveRetangle)):
                        print(self.saveRetangle[i])
                    self.saveRetangle = []
                    self.workStack.clear()
                    break
                elif a == ord('q') or a == ord('Q'):
                    break
                elif a == ord('c') or a == ord('C') or a == ord('N') or a == ord('n'):
                    self.saveRetangle = []
                    self.workStackTemp.clear()
                    image_cv2 = image_cv2_original.copy()
                    self.workStack.push(image_cv2)
                elif a == ord('z') or a == ord('Z'):
                    if self.workStack.length() > 1:
                        self.workStackTemp.push(self.workStack.pop())
                        image_cv2: np.ndarray = self.workStack.top()
                        self.saveRetangle = []
                    else:
                        print("Cannot undo.")
        cv2.destroyAllWindows()
        print("Finish")
        print("Splitting dataset... ")
        split.split("datasets/original", 0.8, 0.1, 0.1, "datasets/split",
                    "D:\\documents\\GitHub\\Image-Train-Tool-for-YOLO\\datasets\\split")


if __name__ == "__main__":
    if sys.argv[1] == "test":
        imageSource = glob.glob("test/*.jpeg")
        imageSource = imageSource + glob.glob("test/*.jpg")
        imageSource = imageSource + glob.glob("test/*.png")
        imageSource = imageSource + glob.glob("test/*.bmp")
        tool = Tool(imageSource)
        tool.Run()
    elif sys.argv[1] == "train":
        # split.split("datasets/original", 0.8, 0.1, 0.1, "datasets/split",
        #             "D:\\documents\\GitHub\\Image-Train-Tool-for-YOLO\\datasets\\split")
        file = open("a.yaml", "w")
        file.write("train: D:\\documents\\GitHub\\Image-Train-Tool-for-YOLO\\datasets\split\\train\\train.txt\n")
        file.write("val: D:\\documents\\GitHub\\Image-Train-Tool-for-YOLO\\datasets\\split\\valid\\valid.txt\n")
        file.write("nc: 1\n")
        file.write("names: \n 0: haibara\n")
        file.close()
        file = open("yolov8.yaml", "r+")
        content = file.read()
        file.close()
        content = content.replace("nc: 80", "nc: 1")
        file = open("yolov8.yaml", "w")
        file.write(content)
        file.close()
        # Load a model
        model = YOLO('yolov8.yaml').load("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='D:\\documents\\GitHub\\Image-Train-Tool-for-YOLO\\a.yaml', epochs=200, imgsz=800,
                              patience=0, device=0, batch=2)
    elif sys.argv[1] == 'detect':
        model = YOLO('runs/detect/train/weights/last.pt')
        image = 'test/8F1CD26C2763D2DDB777EF6DF6636584.jpg'
        img = cv2.imread(image)
        results = model(img)
        for i in results:
            for j in i.boxes:
                xyxy = j.xyxy[0]
                cls = j.cls
                label = results[0].names[int(cls)]
                cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.imshow("Detect", img)
        cv2.waitKey(0)
    elif 'path=' in sys.argv[1]:
        a = sys.argv[1].split('=')
        imageSource = glob.glob(a[1] + "/*.jpeg")
        imageSource = imageSource + glob.glob(a[1] + "/*.jpg")
        imageSource = imageSource + glob.glob(a[1] + "/*.png")
        imageSource = imageSource + glob.glob(a[1] + "/*.bmp")
        tool = Tool(imageSource)
        tool.Run()
    else:
        print("Invalid argument.")
        print("Usage: python main.py [train|test|path=your_path|detect]")
        print("Example: python main.py test")
        print("Example: python main.py train")
        print("Example: python main.py path=test")
        print("Example: python main.py detect")
