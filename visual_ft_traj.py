import numpy as np
import cv2
import skimage.transform as tr
import matplotlib.pyplot as plt

import cv2
import skvideo.io

avg_offsets = []
frame_length = 200

cap = skvideo.io.vreader('data/train/unstable/1.avi')
# 读取图像
frames = []
# cap = cv2.VideoCapture('data/train/unstable/1.avi')
for frame in cap:
    # im = cv2.imread('data/train/stable/1/%d.jpg' % (i + 1))
    im = frame
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    frames.append(im)
# cv2.imshow('original', im)
# cv2.waitKey()
# cap.release()
# 下采样
# im_lowers = cv2.pyrDown(im)
# cv2.imshow('im_lowers',im_lowers)

# 检测特征点
# s = cv2.SIFT() # 调用SIFT
s = cv2.xfeatures2d.SURF_create()  # 调用SURF
for i in range(frame_length):
    keypoints1, des1 = s.detectAndCompute(frames[i], None)
    keypoints2, des2 = s.detectAndCompute(frames[i + 1], None)
    k1 = []
    k2 = []
    for k in keypoints1:
        k1.append((int(k.pt[0]), int(k.pt[1])))
    for k in keypoints2:
        k2.append((int(k.pt[0]), int(k.pt[1])))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    sum_offset = 0
    for m, n in matches:
        md = m.distance
        nd = n.distance
        if m.distance < 0.75 * n.distance:
            good.append([m])
    for p in good:
        x1 = k1[p[0].queryIdx][0]
        y1 = k1[p[0].queryIdx][1]
        x2 = k2[p[0].trainIdx][0]
        y2 = k2[p[0].trainIdx][1]
        # p_offset = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
        p_offset = x2 - x1
        sum_offset += p_offset
    avg_offsets.append(sum_offset / len(good))
    # img3 = cv2.drawMatchesKnn(frames[i], keypoints1, frames[i + 1], keypoints2, good, None, flags=2)

# 显示特征点
# for k in keypoints:
#     k0 = int(k.pt[0])
#     k1 = int(k.pt[1])
#     cv2.circle(im, (k0, k1), 1, (0, 255, 0), -1)
# cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)
plt.plot(np.arange(frame_length), avg_offsets, color='green', lineWidth=0.5)
plt.ylim((-100, 100))
plt.show()
# cv2.imshow('SURF_features', img3)
# cv2.waitKey()
# cv2.destroyAllWindows()
