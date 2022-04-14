import cv2


def convert_one_channel(img):
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def convert_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

# def convert_one_channel(img):
#     if len(img.shape) > 2:
#         return img[:, :, 0]

#     return img
