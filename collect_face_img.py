import os
import cv2
import argparse
from PIL import Image  

cap = cv2.VideoCapture(1) # number of your camera
parser = argparse.ArgumentParser(description='Save face image')
parser.add_argument('person_name', type=str, help='Name of person')
parser.add_argument('ratio', type=int, help='ratio of the frame size')
parser.add_argument('start_index', type=int, help='index/name of the image saved')
args = parser.parse_args()

def collect(person_name, ratio, start_index=1):
    """
    person_name -- string, name of the person
    ratio -- int, width and height of the image will be ratio*64, ratio*48
    start_index -- int, used to continue to collect more images of the same person
    """
    # get PWD
    full_path = os.path.realpath(__file__)
    save_dir = os.path.join(os.path.dirname(full_path), 'data', person_name )
    # /home/mt/Desktop/For_github/computer_vision_projects/face_recognition
    # create folder "person_name" if does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while True:
        # print(save_dir)
        ret, frame = cap.read()
        if ret:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (64*ratio, 48*ratio))
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            
            if key == 32: # space
                # save the image
                cv2.imshow('save', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_save_dir = os.path.join(save_dir, (str(start_index)+'.jpg'))
                print(img_save_dir)
                img = img.save(img_save_dir)
                start_index += 1
                
            elif key == 27:
                break
    cap.release()   
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect(args.person_name, args.ratio, args.start_index)