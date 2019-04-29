import numpy as np
import cv2
import skimage.transform as tr
import matplotlib.pyplot as plt

import cv2

# 读取图像
frames = []
for i in range(20):
    im = cv2.imread('data/train/unstable/1/%d.jpg' % i)
    frames.append(im)
# cv2.imshow('original', im)
# cv2.waitKey()

# 下采样
# im_lowers = cv2.pyrDown(im)
# cv2.imshow('im_lowers',im_lowers)

# 检测特征点
# s = cv2.SIFT() # 调用SIFT
s = cv2.xfeatures2d.SURF_create()  # 调用SURF
for i in range(2 - 1):
    keypoints1, des1 = s.detectAndCompute(frames[i], None)
    keypoints2, des2 = s.detectAndCompute(frames[i + 1], None)
    k1 = []
    k2 = []
    for k in keypoints1:
        k1.append((k.pt[0], k.pt[1]))
    for k in keypoints2:
        k2.append((k.pt[0], k.pt[1]))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(frames[i], keypoints1, frames[i + 1], keypoints2, good[:20], None, flags=2)

# 显示特征点
# for k in keypoints:
#     k0 = int(k.pt[0])
#     k1 = int(k.pt[1])
#     cv2.circle(im, (k0, k1), 1, (0, 255, 0), -1)
    # cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)

cv2.imshow('SURF_features', img3)
cv2.waitKey()
cv2.destroyAllWindows()
