# Face verification

## Using Siamese network

### How to use:

* run collect_face_img.py to save images of someone, make sure images contain only one person, name the folder as the name of the person.
* run the script of detect_face.py. The script will detect person's face and save the image in the same directory.
* The Siamese network load the pre-trained weights and figure out whether input two images are the same person. Or given the network one image, the network will search its database try to find a match, if it failed to find a match, it will output "Non Match".

## To do:
* more images 
* pretrained the model with LFW dataset.
* test out different base network (currently use VGG16)