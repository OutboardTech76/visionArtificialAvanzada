import cv2
import numpy as np
import argparse
from nptyping import NDArray
from typing import Tuple, Any, Union, List

Image = NDArray[(Any, Any, 3), int]


class Tracking:
    """
    Class for handdle tracked objects
    """
    maxFramesUntilInvalid = 10
    minFramesUntilValid = 10
    maxObjectMovement = 80 # In pixels

    def __init__(self, label: str, numFrame: int, x: int, y: int, w: int, h: int ):
        """
        Constructor, sets position and frame when the object appeared
        Args:
            numFrame -> frame when the object appeared
            x, y, w, h -> object position
            confidence
            label
        """
        self.firstFrame = int(numFrame)
        self.outOfFrame = False
        self.framesWithoutDetection = 0
        self.framesDetected = 1
        self.poseHistory = [[int(x), int(y), int(w), int(h)]]
        self.label = label

    def checkFrame(self, detectedPoses, detectedLabels):
        """
        Check if the object is in the frame
        Args:
            detectedPoses -> List of lists of 4 elements with objects position [x,y,w,h]
            detectedLabels -> List with labels
        Returns:
            Index of the object in detectedPoses. -1 if it is not there
        """
        minDist = 90000
        minIndex = -1
        for index,_ in enumerate(detectedPoses):
            xDist = abs(detectedPoses[index][0] - self.poseHistory[-1][0])
            yDist = abs(detectedPoses[index][1] - self.poseHistory[-1][1])
            wDist = abs(detectedPoses[index][2] - self.poseHistory[-1][2])
            hDist = abs(detectedPoses[index][3] - self.poseHistory[-1][3])
            if (xDist and yDist) < Tracking.maxObjectMovement:
                if (wDist and hDist) < Tracking.maxObjectMovement:
                    if self.label == detectedLabels[index]:
                        currentDist1 =  manhattanDist(xDist, wDist)
                        currentDist2 =  manhattanDist(yDist, hDist)
                        currentDist = max(currentDist1, currentDist2)
                        if currentDist < minDist:
                            minIndex = index
                            minDist = currentDist
        return minIndex

    def trackObject(self, detectedPoses: List[int], detectedLabels: List[str], numFrame: int) -> int:
        """
        Method to managa tracked object
        Args:
            detectedPoses -> List of lists of 4 elements with objects position [x,y,w,h]
            detectedLabels -> List with labels
            detectionConfidences -> List with confidences
            numFrame -> Frame number when the detections ocurred
        Returns:
            Index of the object in detectedPoses. -1 if it is not there
        """

        if self.outOfFrame:
            return -1

        index = self.checkFrame(detectedPoses, detectedLabels)

        if index > 0:
            self.framesDetected += 1
            self.framesWithoutDetection = 0
            self.poseHistory.append(detectedPoses[index])
        else:
            self.framesWithoutDetection += 1

        if self.framesWithoutDetection >= Tracking.maxFramesUntilInvalid:
            self.outOfFrame = True
        return index

    def checkObject(self) -> bool:
        """
        Check if the object is valid or not
        Returns:
            True or False
        """
        if self.framesDetected >= Tracking.minFramesUntilValid:
            return True
        return False

def manhattanDist(a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
    c = []
    if type(a) is int or float:
        return (abs(a - b))
    else:
        for i in len(a):
            c[i] = sum(abs(a[i] - b[i]))
        return c

def euclideanDist(a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
    return np.sqrt(sum((abs(a) - abs(b))**2))

def parser():
    """
    Parser function
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default="video.mp4", help="Path to input video. Default: video.mp4")
    ap.add_argument("-w", "--weights", default="yolov3-face.weights", help="YOLO weights. Default: yolov3-face.weights")
    ap.add_argument("-f", "--file", default="yolov3-face.cfg", help="YOLO configuration file (.cfg). Default: yolov3-face.cfg")
    ap.add_argument("-l", "--labels", default="face.names", help="Labels file. Default: faces.names")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections. Default: 0.5")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold when applying non-maxima suppresion. Default: 0.3")
    args = vars(ap.parse_args())
    return args

def getOutputLayers(net):
    """
    Function to gett the ouput layer
    Args: 
        net
    Returns:
        layer name
    """
    layers = net.getLayerNames()
    out = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return out

def drawBoundingBox(img: Image, label: str, confidence: float, color: Tuple[Any, Any, Any],
                    x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Draw box around objects/people and write obbjects name
    Args:
        img -> image where draw
        label -> object name
        confidence -> between 0-1
        color -> random color
        x1, y1 -> initial position box
        x2, y2 -> end position box. x2 = x1 + w; y2 = y1 + h
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, "{} ({}%)".format(label, round(confidence*100)), (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def showNumObj(img: Image, faces: int) -> None:
    """
    Shows number of faces in every frame
    Args:
        img
        face -> numFaces
    """
    cv2.putText(img, "Current frame:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
    cv2.putText(img, "Faces: {}".format(faces), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )

def showTotalObj(img: Image, faces: int) -> None:
    """
    Shows number of faces
    Args:
        img
        face -> total num Faces
    """
    cv2.putText(img, "Total:", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
    cv2.putText(img, "Faces: {}".format(faces), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )


     

if __name__ == '__main__':
    args = parser()
    cap = cv2.VideoCapture(args["video"])

    classes = None
    with open(args["labels"], 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    np.random.seed(30)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    height = cap.get(4)
    width = cap.get(3)

    net = cv2.dnn.readNet(args["weights"], args["file"])
    trackedObjectList = []
    numFrame = 0
    counter = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        classIDs = []
        confidences = []
        boxes = []
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(getOutputLayers(net))
        numFaces = 0

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (xCenter, yCenter, w, h) = box.astype("int")
                    x = int(xCenter - w / 2)
                    y = int(yCenter - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    classIDs.append(classID)
                    confidences.append(float(confidence))
                     
         # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        for i in indices:
            i = i[0]
            x, y, w, h = boxes[i]
            indx = classIDs[i]
            label = str(classes[indx])
            drawBoundingBox(frame, label, confidences[i], colors[indx], round(x),round(y), round(x+w), round(y+h))
            if label == "face":
                numFaces += 1
                 
        showNumObj(frame, numFaces)

        detectedLabels = [classes[classIDs[i[0]]] for i in indices]
        detectedPoses = [boxes[i[0]] for i in indices]

        for idx,_ in enumerate(trackedObjectList):
            indexFound = trackedObjectList[idx].trackObject(detectedPoses, detectedLabels, numFrame)
            if indexFound > 0:
                del detectedPoses[indexFound]
                del detectedLabels[indexFound]
        for pos, label in zip(detectedPoses, detectedLabels):
            trackedObjectList.append(Tracking(label, numFrame, pos[0], pos[1], pos[2], pos[3]))
         
        numFrame += 1
         
         
        totalFaces = sum([1 if obj.checkObject() and obj.label == "face" else 0 for obj in trackedObjectList])
        showTotalObj(frame, totalFaces)
        cv2.imshow('a',frame)
         
         
         
         
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

