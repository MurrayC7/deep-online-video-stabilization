import pickle
from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import os
import numpy as np
from numpy.linalg import inv
from PWCNet.PyTorch.script_pwc import calcOpticalFlowPWC


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


# frame_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/stable/1/"
#
# video_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/unstable/1.avi"
# gen_video_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/unstable/1gen.avi"
# gen_frame_path = "/Users/plusub/PycharmProjects/deep-online-video-stabilization/data/train/unstable/1/"

# hyperparameters
rho = 8
patch_size = 200
height = 240
width = 320
visualize = False
num_examples = 1000

indices = [0, 1, 2, 4, 8, 16, 32]
gen_batch_size = 30


def gen_with_unstable_flow(cap, stable_frame_path, frame_save_path):
    loc_list = glob(os.path.join(stable_frame_path, '*.jpg'))
    loc_list.sort(key=lambda x: int(x[len(stable_frame_path):-4]))
    # Take first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    prev_frame, prev_gray = first_frame, first_gray

    gen_idx = 0
    last_flow = []
    ret = True
    while ret:
        # processing frames
        ret, next_frame = cap.read()
        if ret:

            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            # os.system("python2 /media/omnisky/cc/PWC-Net/PyTorch/script_pwc.py '%s' '%s' '%s'" % (prev_framepath, framepath, ofpath))
            flow = calcOpticalFlowPWC(prev_gray, next_gray)
            # flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            last_flow = flow
        else:
            flow = last_flow
        # calculate mat and warp
        img_file_location = loc_list[gen_idx]
        stable_frame = plt.imread(img_file_location)
        h, w, c = stable_frame.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow

        # preserve aspect ratio
        global HORIZONTAL_BORDER
        HORIZONTAL_BORDER = 30

        global VERTICAL_BORDER
        VERTICAL_BORDER = (HORIZONTAL_BORDER * w) // h

        new_frame = cv2.remap(stable_frame, pixel_map, None, cv2.INTER_LINEAR)
        new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
        # new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        # print(gen_idx)
        cv2.imwrite(os.path.join(frame_save_path, "%d.jpg" % (gen_idx + 1)), new_frame)
        if ret:
            prev_gray = next_gray
        else:
            continue
        gen_idx += 1


def gen_frame():
    loc_list = glob(os.path.join(frame_path, '*.jpg'))
    loc_list.sort(key=lambda x: int(x[len(frame_path):-4]))
    # X = np.zeros((128, 128, 2, num_examples))  # images
    # Y = np.zeros((4, 2, num_examples))

    num_batch = int(len(loc_list) / gen_batch_size) + 1
    last_batch_size = len(loc_list) - gen_batch_size * (num_batch - 1)
    for idx in range(num_batch):
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

        if idx == num_batch - 1:
            batch_size = last_batch_size
        else:
            batch_size = gen_batch_size

        for i in range(batch_size):
            img_file_location = loc_list[idx * gen_batch_size + i]
            color_image = plt.imread(img_file_location)
            h, w, c = color_image.shape
            # preserve aspect ratio
            global HORIZONTAL_BORDER
            HORIZONTAL_BORDER = 30

            global VERTICAL_BORDER
            VERTICAL_BORDER = (HORIZONTAL_BORDER * w) // h
            # color_image = cv2.resize(color_image, (width, height))
            # gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            gray_image = color_image

            # compute H
            H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            H_inverse = inv(H)
            inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (w, h))
            warped_image = cv2.warpPerspective(gray_image, H, (w, h))
            new_frame = warped_image[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
            # new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(os.path.join(gen_frame_path, "%d.jpg" % (idx * gen_batch_size + i + 1)), new_frame)
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
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(save_dir, fourcc, fps, (1174, 660))
    frames = glob(os.path.join(frame_dir, "*jpg"))
    frames.sort(key=lambda x: int(x[len(frame_dir):-4]))
    for i in range(len(frames)):
        frame = cv2.imread(frames[i])
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    # gen_frame()
    videos_src_path = '../DeepStab/unstable'
    videos_save_path = './data_video/gen_unstable/video'
    frame_save_path = './data_video/gen_unstable/frame'
    stable_frame_path = './data_video/stable'

    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('avi'), videos)

    for each_video in videos:
        print(each_video)

        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.mkdir(frame_save_path + '/' + each_video_name)
        os.mkdir(videos_save_path + '/' + each_video_name)

        each_frame_save_full_path = os.path.join(frame_save_path, each_video_name) + '/'
        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'
        each_stable_frame_full_path = os.path.join(stable_frame_path, each_video_name) + '/'

        # get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        gen_with_unstable_flow(cap, each_stable_frame_full_path, each_frame_save_full_path)
        frame2video(each_frame_save_full_path, each_video_save_full_path)
    print("done")
    print("")
