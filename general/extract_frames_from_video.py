import cv2
from matplotlib import pyplot as plt

video_dir = '/home/nearlab/Jorge/data/ureteroscopy/' \
            'data_phantoms/phantom_003/'
name_video = 'phtantom_003_pt01'

dir_video = ''.join([video_dir, name_video, '.mp4'])

save_directory = '/home/nearlab/Jorge/data/ureteroscopy/data_phantoms/phantom_003/frames_pt01/'

print(dir_video)
vidcap = cv2.VideoCapture(dir_video)
success, image = vidcap.read()
print(success)
count = 0

while success:
    save_name = ''.join([save_directory, name_video, '_', '{:0>4}'.format(count), ".png"])
    """print(count, save_name)
    plt.figure()
    plt.imshow(image)
    plt.show()"""
    cv2.imwrite(save_name, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1