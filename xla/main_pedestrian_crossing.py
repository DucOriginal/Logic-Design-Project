from Lane_detection import lane_finding_pipeline, road_markings
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def image(input_path):
    for image in sorted(os.listdir(input_path)):
        if(image.endswith(('.png', '.jpg'))):
            #get the list of images in the test folder
            # img_list = os.listdir(image)
            path = os.path.join(input_path, image)
            img = cv2.imread(path)
            img_out = road_markings(img)
            cv2.imshow('img_out', img_out)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                
if __name__ == '__main__':
    image_Path= './road_maker'
    image(image_Path)