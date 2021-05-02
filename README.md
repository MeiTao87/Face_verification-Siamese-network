# Face verification

## Using Siamese network

### How to use:

* save images of someone, make sure images contain only one person, name the folder as the name of the person.
* run the script of collect_face_img and detect_face. These two scripts will collect detect and crop images which only contain person's face.
* The Siamese network load the pre-trained weights and figure out whether input two images are the same person. Or given the network one image, the network will search its database try to find a match, if it failed to find a match, it will output a "Non Match".