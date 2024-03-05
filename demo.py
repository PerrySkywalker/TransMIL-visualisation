import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2
if __name__ == "__main__":
    tree = ET.parse('xml/test_040.xml')
    img = cv2.imread('out/out1/test_040.jpg')
    root = tree.getroot()
    for Annotations in root.findall('Annotations'):
        for Annotation in Annotations.findall('Annotation'):
            closedCurves = []
            for Coordinates in Annotation.findall('Coordinates'):
                for Coordinate in Coordinates.findall("Coordinate"):
                    closedCurves.append([int(float(Coordinate.attrib["X"])/64), int(float(Coordinate.attrib["Y"])/64)])
                points = np.array(closedCurves, dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(img, [points], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.imwrite('demo.jpg', img)
