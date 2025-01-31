import os
import cv2


def main():
    videos_src_path = '../DeepStab/stable'
    videos_save_path = './data_video/stable'

    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('avi'), videos)

    for each_video in videos:
        print(each_video)

        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.mkdir(videos_save_path + '/' + each_video_name)

        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

        # get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            if not success:
                print('Read a new frame: ', success, frame_count)

            params = []
            params.append(cv2.IMWRITE_PXM_BINARY)
            params.append(1)
            cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, frame, params)

            frame_count = frame_count + 1

    cap.release()


def rename_():
    basedir = "./data/train/unstable"
    for root, subs, files in os.walk(basedir):
        # print("root:", root, files, subs, len(subs))
        for sub in subs:

            path = os.path.join(basedir, sub)
            prefix_len = len(sub)
            # print("path:", path, len(sub))
            for file in os.listdir(path):
                if "txt" in file:
                    # print("file:", file)
                    src = os.path.join(path, file)
                    dst = os.path.join(path, file[prefix_len:])
                    print("src:", src, "---------dst:", dst)
                    os.rename(src, dst)
                else:
                    continue


if __name__ == '__main__':
    main()
