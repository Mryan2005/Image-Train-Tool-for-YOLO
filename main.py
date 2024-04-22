import glob
import sys

import cv2
import numpy as np


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item: np.ndarray):
        self.stack.append(item)

    def pop(self) -> np.ndarray:
        if len(self.stack) == 0:
            return np.array([])
        return self.stack.pop(-1)

    def top(self) -> np.ndarray:
        if len(self.stack) == 0:
            return np.array([])
        return self.stack[-1]

    def clear(self):
        self.stack = []


class Tool:
    def __init__(self, imageSource: list):
        self.workStack = Stack()
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
                    self.saveRetangle.append((self.selectRetangle))
                    ima = image_cv2.copy()
                    self.workStack.push(ima)
                    self.selectRetangle = []
                cv2.imshow(self.windowName, image_cv2)  # type: ignore
                a = cv2.waitKey(1)
                if a == ord('Y') or a == ord('y'):
                    for i in range(len(self.saveRetangle)):
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
                    self.workStack.clear()
                    self.saveRetangle = []
                    image_cv2 = image_cv2_original.copy()
                    self.workStack.push(image_cv2.copy())
                elif a == ord('z') or a == ord('Z'):
                    if self.workStack.stack.__len__() > 1:
                        self.workStack.pop()
                        image_cv2: np.ndarray = self.workStack.top().copy()
                        self.saveRetangle.pop()
                    else:
                        print("Cannot undo.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.argv[1] == "test":
        imageSource = glob.glob("test/*.jpeg")
        imageSource = imageSource + glob.glob("test/*.jpg")
        imageSource = imageSource + glob.glob("test/*.png")
        imageSource = imageSource + glob.glob("test/*.bmp")
        tool = Tool(imageSource)
        tool.Run()
