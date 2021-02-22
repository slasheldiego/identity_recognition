from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import asarray
from numpy import expand_dims

import cv2
import numpy as np
 
# get faces from images
def get_face(filename, haar_path, output_face_size=(160, 160)):
    # load image from file
    image = cv2.imread(filename)
    # convert to RGB. Normally OpenCV load image in BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # create the detector, using Viola & Jones detector
    detector = cv2.CascadeClassifier(haar_path)
    # detect faces 
    rects = detector.detectMultiScale(image, scaleFactor=1.3,minNeighbors=5, minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) > 0:
        rect = sorted(rects, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        # extract the bounding box from the first face
        x1, y1, width, height = rect
        # extract the face
        face = image[y1:y1 + height, x1:x1 + width]
        # resize image to the model size
        face = cv2.resize(face,output_face_size)
    else:
        face = None
    
    return face

# load images and extract faces for all images in a directory
def load_faces(directory,haar_path):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = get_face(filename=path,haar_path=haar_path)
        
        if face is not None:
            # store
            faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, haar_path):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(directory=path,haar_path=haar_path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# get facenet feature for a face
def get_facenet_feature(model, faces):
    # Parce pixel values type to float
    faces = faces.astype('float32')
    # standardize pixel values (normalization)
    mean, std = faces.mean(), faces.std()
    faces = (faces - mean) / std
    # transform face into one sample
    samples = expand_dims(faces, axis=0)
    # use de model to get de facenet feature (using predict module)
    yhat = model.predict(samples)
    return yhat[0]