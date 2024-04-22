import cv2
import numpy as np
import os
import sys
import glob

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
                    image_cv2 = cv2.rectangle(image_cv2, self.selectRetangle[0], self.selectRetangle[1], (0, 255, 0), 2) # type: ignore
                    self.saveRetangle.append((self.selectRetangle))
                    ima = image_cv2.copy()
                    self.workStack.push(ima)
                    self.selectRetangle = []
                cv2.imshow(self.windowName, image_cv2) # type: ignore
                a = cv2.waitKey(1)
                if a == ord('c'):
                    cv2.imwrite("datasets/original/images/" + str(self.count) + ".jpeg", image_cv2_original)
                    file = open("datasets/original/labels/" + str(self.count) + ".txt", "w")
                    file.write('0 ')
                    for i in range(len(self.saveRetangle)):
                        file.write(str(self.saveRetangle[self.count-1][0][0]/800) + ' ' + str(self.saveRetangle[self.count-1][0][1]/800) + ' ' + str(self.saveRetangle[self.count-1][1][0]/800) + ' ' + str(self.saveRetangle[self.count-1][1][1]/800) + '\n')
                    print("Save image: ", "image/" + str(self.count) + ".jpeg")
                    self.saveRetangle = []
                    for i in range(len(self.saveRetangle)):
                        print(self.saveRetangle[i])
                    self.workStack.clear()
                    break
                elif a == ord('q'):
                    break
                elif a == ord('z'):
                    if self.workStack.stack.__len__() > 1:
                        self.workStack.pop()
                        image_cv2: np.ndarray = self.workStack.top().copy()
                    else:
                        print("Cannot undo.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        if sys.argv[1] == "test":
            imageSource = glob.glob("test/*.jpeg")
            imageSource = imageSource + glob.glob("test/*.jpg")
            imageSource = imageSource + glob.glob("test/*.png")
            imageSource = imageSource + glob.glob("test/*.bmp")
            tool = Tool(imageSource)
            tool.Run()
    except IndexError:
        if 'path=' in sys.argv[1]:
            imageSource = glob.glob(sys.argv[1].split("path=")+"/*.jpeg")
            tool = Tool(imageSource)
            tool.Run()
        else:
            print("Please input the path of the image folder.")
    except Exception as e:
        print(e)
        print("Please input the path of the image folder.")