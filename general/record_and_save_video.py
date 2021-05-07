# Python program to illustrate
# saving an operated video

# organize imports
import numpy as np
import cv2

# This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))
save_dir = '/home/nearlab/Jorge/data/ureteroscopy/data_phantoms/' \
           'phantom_007/frames_pt01/'

# loop runs if capturing has been initialized.
img_id = 0000
while (True):
    # reads frames from a camera
    # ret checks return at each frame
    ret, frame = cap.read()

    # Converts to HSV color space, OCV reads colors as BGR
    # frame is converted to hsv
    resized = cv2.resize(frame, (300, 300))


    # The original input frame is shown in the window
    cv2.imshow('Original', resized)
    img_name = 'phantom_007_pt01_' + str(img_id).zfill(4) + '.png'
    print(img_name)
    cv2.imwrite(save_dir + img_name, resized)
    img_id = img_id + 1
    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
out.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()