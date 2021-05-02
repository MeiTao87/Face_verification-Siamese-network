import cv2
import os

def detect_face_save(img_path):
    for sub_folder in os.listdir(img_path):
        if os.path.isdir(os.path.join(img_path, sub_folder)): # if it is a directory
            for img in os.listdir(os.path.join(img_path, sub_folder)):
                if img.endswith('.jpg'):
                    new_file_name = img[:-4] + '_face' + '.jpg'
                    img_whole_path = os.path.join(img_path, sub_folder, img)
                    new_img_path = os.path.join(img_path, sub_folder, img[:-4] + 'face' + '.jpg')
                    image = cv2.imread(img_whole_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=3,
                        minSize=(15, 15)
                    )
                    for (x, y, w, h) in faces:
                        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = image[y:y + h, x:x + w]
                        status = cv2.imwrite(new_img_path, roi_color)
                        print(f"{new_img_path} written to disc: ", status)

if __name__ == '__main__':
    #imagePath contains the image from which the face needs to be extracted
    imagePath = './data/'
    detect_face_save(imagePath)