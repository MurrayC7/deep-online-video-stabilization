import pickle
from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import os
import numpy as np
from numpy.linalg import inv


def show_image(location, title, img, width=None):
    if width is not None:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if width is not None:
        plt.show()
        plt.close()


video_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/stable/1.avi"
frame_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/stable/1/"
gen_video_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/unstable/1.avi"
gen_frame_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/unstable/1/"

# hyperparameters
rho = 8
patch_size = 128
height = 240
width = 320
visualize = False
num_examples = 1000

loc_list = glob(os.path.join(frame_path, '*.jpg'))
# X = np.zeros((128, 128, 2, num_examples))  # images
# Y = np.zeros((4, 2, num_examples))
for i in range(len(frame_path)):
    img_file_location = loc_list[i]
    color_image = plt.imread(img_file_location)
    w, h, _ = color_image.shape
    B = color_image[:, :, 0]
    G = color_image[:, :, 1]
    R = color_image[:, :, 2]
    color_image = cv2.resize(color_image, (width, height))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # create random point P within appropriate bounds
    y = random.randint(rho, height - rho - patch_size)  # row?
    x = random.randint(rho, width - rho - patch_size)  # col?
    # define corners of image patch
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
    warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

    gen_image = cv2.resize(warped_image, (w, h))
    gen_image[:, :, 0] = B
    gen_image[:, :, 1] = G
    gen_image[:, :, 2] = R
    cv2.imwrite(os.path.join(gen_frame_path, "%d.jpg" % (i + 1)), warped_image)
    # # grab image patches
    # original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    # warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    # # make into dataset
    # training_image = np.dstack((original_patch, warped_patch))
    # H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    # X[:, :, :, i] = training_image
    # Y[:, :, i] = H_four_points


def frame2video(frame_dir, save_dir):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_dir, fourcc, fps, (1280, 720))
    frames = glob(os.path.join(frame_dir, "*jpg"))
    for i in range(len(frames)):
        frame = cv2.imread(frames[i])
        video_writer.write(frame)
    video_writer.release()


frame2video(gen_frame_path, gen_video_path)
