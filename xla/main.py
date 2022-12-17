from Lane_detection import lane_finding_pipeline
import os
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

def image(input_path):
    for image in sorted(os.listdir(input_path)):
        if(image.endswith(('.png', '.jpg'))):
            #get the list of images in the test folder
            img_list = os.listdir("test_images/")

            img = mpimg.imread('test_images/solidWhiteRight.jpg')
            img_out = lane_finding_pipeline(img)
            cv2.imshow('img_out', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
def video(path_video):
    cap = cv2.VideoCapture(path_video)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
       
    # size = (frame_width, frame_height)
       
    # # Below VideoWriter object will create
    # # a frame of above defined The output 
    # # is stored in 'filename.avi' file.
    # result = cv2.VideoWriter('ouput.avi', 
    #                          cv2.VideoWriter_fourcc(*'MJPG'),
    #                          10, size)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # inference
            # try:
            if True:
                img_out = lane_finding_pipeline(frame)
                # result.write(img_out)
                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow('window',img_out)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # except Exception as e:
            #     print(e)
            # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()
if __name__ == '__main__':
    path_video1='./test_videos/1.mp4'
    
    path_video2= './test_videos/solidWhiteRight.mp4'
    video(path_video2)