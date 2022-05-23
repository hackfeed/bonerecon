import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
from sklearn import svm

MOLARES_COLOR = [255, 0, 0]
PREMOLARES_COLOR = [0, 255, 0]
CANINOS_COLOR = [255, 255, 0]
INCISIVOS_COLOR = [0, 0, 255]

COLORS = {
    0: MOLARES_COLOR,
    1: PREMOLARES_COLOR,
    2: CANINOS_COLOR,
    3: INCISIVOS_COLOR
}


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def classifier():
    X = [
        # Molares
        [286.9, 153.1],
        [289.6, 142.5],
        [236.8, 134.7],
        [272.4, 153.2],
        [287.9, 150.1],
        [290.9, 133],
        [274.1, 136.5],
        [195.5, 129.6],
        [310.6, 139.2],
        [263.4, 149],
        [230.7, 159.8],

        [338.3, 144.5],
        [309.5, 137.3],
        [277.6, 149.5],
        [365.9, 153.1],
        [375.7, 147.6],
        [351.4, 113.6],
        [256, 133],
        [283.3, 182.2],
        [368.4, 153.7],
        # Premolares
        [281.4, 70.6],
        [284.5, 79.7],
        [285.3, 89.1],
        [289.5, 83.4],
        [273.5, 81.2],
        [289.6, 77.4],
        [290.3, 75.8],
        [282, 75],
        # Caninos
        [406, 99],
        [384.3, 96],
        [377.1, 83],
        [363.5, 82.1],
        [433.6, 111],
        [350.7, 88.4],
        [388.8, 101.1],
        [363.9, 91.2],
        # Incisivios
        [327.5, 87.6],
        [360, 104],
        [327.2, 103],
        [331.6, 86.5],
        [314, 60],
        [288, 60],
        [280, 64],
        [314.3, 62.1],
    ]

    Y = [
        # Molares
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # Premolares
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        # Caninos
        2,
        2,
        2,
        2,

        2,
        2,
        2,
        2,
        # Incisivios
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]

    return svm.SVC().fit(X, Y)


def analyze(orig_image, predict_image, erode_iteration, open_iteration, use_svm=False):
    clf = None
    if use_svm:
        clf = classifier()

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

        if use_svm:
            tooth = rect[1]
            if tooth[1] > tooth[0]:
                tooth = [tooth[1], tooth[0]]
            color = COLORS[clf.predict([tooth])[0]]
        else:
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
