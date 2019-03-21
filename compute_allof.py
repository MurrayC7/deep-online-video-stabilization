import os

frame_folder = '/media/omnisky/cc/coupe.DVSG/images/unstable/1/'
framenames = [f for f in os.listdir(frame_folder) if f.endswith('.ppm')]
framenames.sort(key=lambda x:int(x[:-4]))
prev_framepath = frame_folder + framenames[0]
first_frame = 1
for framename in framenames:
    print("frame loc(prev, cur):", prev_framepath.split('/')[-1], framename)
    if 'ppm' in framename and first_frame != 1:
        framepath = frame_folder + framename
        ofpath = frame_folder + 'of/' + prev_framepath.split('/')[-1].split('.')[0] + '.flo'
        #print("'prev: %s' 'cur: %s' 'of: %s'" % (prev_framepath, framepath, ofpath))
        os.system("python2 /media/omnisky/cc/PWC-Net/PyTorch/script_pwc.py '%s' '%s' '%s'" % (prev_framepath, framepath, ofpath))
    else:
        first_frame = 0
        continue
    prev_framepath = framepath
