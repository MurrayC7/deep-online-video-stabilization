import imutils
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from tqdm import tqdm

video_path = './data/train/unstable/0.un_cut.mp4'
PIXELS = 16
RADIUS = 300
HORIZONTAL_BORDER = 30


def get_outline(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)
    _, RedThresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def findContours_img(origin_img, opened):
    image, cnts, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print("cnts:", cnts, np.shape(cnts))
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # print(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    draw_img = cv2.drawContours(origin_img.copy(), cnts, -1, (0, 0, 255), 3)
    print('box0', box[0])
    print('box0', box[1])
    print('box0', box[2])
    print('box0', box[3])
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt, box, draw_img, cnts


def find_H_and_area(img):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opened = get_outline(gray)
    docCnt, box, draw_img, cnts = findContours_img(img, opened)

    crop_area = areaCal(cnts)

    dst_point = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    h, s = cv2.findHomography(docCnt, dst_point, cv2.RANSAC, 10)
    print(h)
    cv2.imshow('opened', opened)
    # cv2.imshow('box', box)
    cv2.imshow('draw_img', draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return h, crop_area


def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])
        print('area(i):', area)
    return area


def cropping_ratio(cap):
    for i in range(100):
        ret, frame = cap.read()
    _, crop_area = find_H_and_area(frame)
    total_area = frame.shape[0] * frame.shape[1]
    crop_ratio = crop_area / total_area
    return crop_ratio


def distortion_value(cap):
    for i in range(100):
        ret, frame = cap.read()
    h, _ = find_H_and_area(frame)
    vals, vecs = np.linalg.eig(h)
    vals = abs(vals)
    print(vals)
    vals.sort()
    print(vals)
    dis_value = (vals[0] + vals[1]) / np.sum(vals)
    return dis_value


def stability_score(cap):
    def point_transform(H, pt):
        """
        @param: H is homography matrix of dimension (3x3)
        @param: pt is the (x, y) point to be transformed

        Return:
                returns a transformed point ptrans = H*pt.
        """
        a = H[0, 0] * pt[0] + H[0, 1] * pt[1] + H[0, 2]
        b = H[1, 0] * pt[0] + H[1, 1] * pt[1] + H[1, 2]
        c = H[2, 0] * pt[0] + H[2, 1] * pt[1] + H[2, 2]
        return [a / c, b / c]

    def motion_propagate(old_points, new_points, old_frame):
        """
        @param: old_points are points in old_frame that are
                matched feature points with new_frame
        @param: new_points are points in new_frame that are
                matched feature points with old_frame
        @param: old_frame is the frame to which
                motion mesh needs to be obtained
        @param: H is the homography between old and new points

        Return:
                returns a motion mesh in x-direction
                and y-direction for old_frame
        """
        # spreads motion over the mesh for the old_frame
        x_motion = {};
        y_motion = {};
        cols, rows = old_frame.shape[1] / PIXELS, old_frame.shape[0] / PIXELS

        # pre-warping with global homography
        H, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
        for i in range(rows):
            for j in range(cols):
                pt = [PIXELS * j, PIXELS * i]
                ptrans = point_transform(H, pt)
                x_motion[i, j] = pt[0] - ptrans[0]
                y_motion[i, j] = pt[1] - ptrans[1]

        # disturbute feature motion vectors
        temp_x_motion = {};
        temp_y_motion = {}
        for i in range(rows):
            for j in range(cols):
                vertex = [PIXELS * j, PIXELS * i]
                for pt, st in zip(old_points, new_points):

                    # velocity = point - feature point match in next frame
                    # dst = sqrt((vertex[0]-st[0])**2+(vertex[1]-st[1])**2)

                    # velocity = point - feature point in current frame
                    dst = np.sqrt((vertex[0] - pt[0]) ** 2 + (vertex[1] - pt[1]) ** 2)
                    if dst < RADIUS:
                        ptrans = point_transform(H, pt)
                        try:
                            temp_x_motion[i, j].append(st[0] - ptrans[0])
                        except:
                            temp_x_motion[i, j] = [st[0] - ptrans[0]]
                        try:
                            temp_y_motion[i, j].append(st[1] - ptrans[1])
                        except:
                            temp_y_motion[i, j] = [st[1] - ptrans[1]]

        # apply median filter (f-1) on obtained motion for each vertex
        x_motion_mesh = np.zeros((rows, cols), dtype=float)
        y_motion_mesh = np.zeros((rows, cols), dtype=float)
        for key in x_motion.keys():
            try:
                temp_x_motion[key].sort()
                x_motion_mesh[key] = x_motion[key] + temp_x_motion[key][len(temp_x_motion[key]) / 2]
            except KeyError:
                x_motion_mesh[key] = x_motion[key]
            try:
                temp_y_motion[key].sort()
                y_motion_mesh[key] = y_motion[key] + temp_y_motion[key][len(temp_y_motion[key]) / 2]
            except KeyError:
                y_motion_mesh[key] = y_motion[key]

        # apply second median filter (f-2) over the motion mesh for outliers
        x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
        y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])

        return x_motion_mesh, y_motion_mesh

    def generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh):
        """
        @param: x_paths is vertex profiles along x-direction
        @param: y_paths is vertex profiles along y_direction
        @param: x_motion_mesh is obtained motion mesh along
                x-direction from motion_propogate()
        @param: y_motion_mesh is obtained motion mesh along
                y-direction from motion_propogate()

        Returns:
                returns updated x_paths, y_paths with new
                x_motion_mesh, y_motion_mesh added to the
                last x_paths, y_paths
        """
        new_x_path = x_paths[:, :, -1] + x_motion_mesh
        new_y_path = y_paths[:, :, -1] + y_motion_mesh
        x_paths = np.concatenate((x_paths, np.expand_dims(new_x_path, axis=2)), axis=2)
        y_paths = np.concatenate((y_paths, np.expand_dims(new_y_path, axis=2)), axis=2)
        return x_paths, y_paths

    def motion_mesh():
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=1000,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        # Take first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # motion meshes in x-direction and y-direction
        x_motion_meshes = [];
        y_motion_meshes = []

        # path parameters
        x_paths = np.zeros((old_frame.shape[0] / PIXELS, old_frame.shape[1] / PIXELS, 1))
        y_paths = np.zeros((old_frame.shape[0] / PIXELS, old_frame.shape[1] / PIXELS, 1))

        frame_num = 1
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        bar = tqdm(total=frame_count)
        while frame_num < frame_count:

            # processing frames
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find corners in it
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # estimate motion mesh for old_frame
            x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
            try:
                x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
                y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
            except:
                x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
                y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

            # generate vertex profiles
            x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

            # updates frames
            bar.update(1)
            frame_num += 1
            old_frame = frame.copy()
            old_gray = frame_gray.copy()

        bar.close()

    pass


if __name__ == '__main__':
    # video_path = ''
    cap = cv2.VideoCapture(video_path)
    crop_ratio = cropping_ratio(cap)
    dis_value = distortion_value(cap)
    print(crop_ratio, dis_value)
    print("done!")
