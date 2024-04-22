import cv2
import numpy as np
import os
import sys
import glob

class Tool:
    def __init__(self, imageSource: list):
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
            image_cv2 = cv2.imread(image)
            image_cv2_copy = image_cv2.copy()
            self.count += 1
            if image_cv2 is None:
                print("Cannot read image: ", image)
                continue
            image_cv2 = cv2.resize(image_cv2, (int(800), int(800)))
            while True:
                if len(self.selectRetangle) == 2:
                    cv2.rectangle(image_cv2, self.selectRetangle[0], self.selectRetangle[1], (0, 255, 0), 2)
                    self.saveRetangle.append((self.selectRetangle))
                    self.selectRetangle = []
                cv2.imshow(self.windowName, image_cv2)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    cv2.imwrite("datasets/original/images/" + str(self.count) + ".jpeg", image_cv2_copy)
                    file = open("datasets/original/labels/" + str(self.count) + ".txt", "w")
                    file.write('0 ')
                    center = ((self.saveRetangle[self.count-1][0][0] + self.saveRetangle[self.count-1][1][0]) / 2, (self.saveRetangle[self.count-1][0][1] + self.saveRetangle[self.count-1][1][1]) / 2)
                    file.write(str(center[0] / 800) + ' ' + str(center[1] / 800) + ' ' + str((self.saveRetangle[self.count-1][1][0] - self.saveRetangle[self.count-1][0][0]) / 800) + ' ' + str((self.saveRetangle[self.count-1][1][1] - self.saveRetangle[self.count-1][0][1]) / 800))
                    file.write('\n')
                    file.close()
                    print("Save image: ", "image/" + str(self.count) + ".jpeg")
                    for i in range(len(self.saveRetangle)):
                        print(self.saveRetangle[self.count-1])
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        if sys.argv[1] == "test":
            imageSource = glob.glob("test/*.jpeg")
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