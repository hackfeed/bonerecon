import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def analyze(orig_image, predict_image, erode_iteration, open_iteration):
    kernel = (np.ones((5, 5), dtype=np.float32))
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                 [-1, -1, -1]])
    pimage = predict_image
    oimage = orig_image

    pimage = cv2.morphologyEx(pimage, cv2.MORPH_OPEN, kernel, iterations=open_iteration)
    pimage = cv2.filter2D(pimage, -1, kernel_sharpening)
    pimage = cv2.erode(pimage, kernel, iterations=erode_iteration)
    pimage = cv2.cvtColor(pimage, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(pimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    labels = cv2.connectedComponents(thresh, connectivity=8)[1]

    count = 0

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue

        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        c_area = cv2.contourArea(cnts)

        if c_area > 2000:
            count += 1

        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        intcolor = (list(np.random.choice(range(150), size=3)))
        color = [int(intcolor[0]), int(intcolor[1]), int(intcolor[2])]

        cv2.drawContours(oimage, [box.astype("int")], 0, color, 2)
        (tl, tr, br, bl) = box

        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(oimage, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(oimage, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(oimage, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(oimage, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(oimage, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), color, 2)
        cv2.line(oimage, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), color, 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        pixelsPerMetric = 1

        dimA = dA * pixelsPerMetric
        dimB = dB * pixelsPerMetric

        cv2.putText(oimage, "{:.1f}pixel".format(dimA), (int(tltrX - 15),
                    int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(oimage, "{:.1f}pixel".format(dimB), (int(trbrX + 10),
                    int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(oimage, "{:.1f}".format(label), (int(tltrX - 35),
                    int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return oimage, count
