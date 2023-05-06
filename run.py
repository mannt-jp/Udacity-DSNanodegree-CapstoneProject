from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import streamlit as st
from extract_bottleneck_features import *
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import ResNet50
import keras.utils as image
import time
from tqdm import tqdm
import json
import pandas as pd
import cv2
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
dog_names = ['Affenpinscher',
            'Afghan hound',
            'Airedale terrier',
            'Akita',
            'Alaskan malamute',
            'American eskimo dog',
            'American foxhound',
            'American staffordshire terrier',
            'American water spaniel',
            'Anatolian shepherd dog',
            'Australian cattle dog',
            'Australian shepherd',
            'Australian terrier',
            'Basenji',
            'Basset hound',
            'Beagle',
            'Bearded collie',
            'Beauceron',
            'Bedlington terrier',
            'Belgian malinois',
            'Belgian sheepdog',
            'Belgian tervuren',
            'Bernese mountain dog',
            'Bichon frise',
            'Black and tan coonhound',
            'Black russian terrier',
            'Bloodhound',
            'Bluetick coonhound',
            'Border collie',
            'Border terrier',
            'Borzoi',
            'Boston terrier',
            'Bouvier des flandres',
            'Boxer',
            'Boykin spaniel',
            'Briard',
            'Brittany',
            'Brussels griffon',
            'Bull terrier',
            'Bulldog',
            'Bullmastiff',
            'Cairn terrier',
            'Canaan dog',
            'Cane corso',
            'Cardigan welsh corgi',
            'Cavalier king charles spaniel',
            'Chesapeake bay retriever',
            'Chihuahua',
            'Chinese crested',
            'Chinese shar-pei',
            'Chow chow',
            'Clumber spaniel',
            'Cocker spaniel',
            'Collie',
            'Curly-coated retriever',
            'Dachshund',
            'Dalmatian',
            'Dandie dinmont terrier',
            'Doberman pinscher',
            'Dogue de bordeaux',
            'English cocker spaniel',
            'English setter',
            'English springer spaniel',
            'English toy spaniel',
            'Entlebucher mountain dog',
            'Field spaniel',
            'Finnish spitz',
            'Flat-coated retriever',
            'French bulldog',
            'German pinscher',
            'German shepherd dog',
            'German shorthaired pointer',
            'German wirehaired pointer',
            'Giant schnauzer',
            'Glen of imaal terrier',
            'Golden retriever',
            'Gordon setter',
            'Great dane',
            'Great pyrenees',
            'Greater swiss mountain dog',
            'Greyhound',
            'Havanese',
            'Ibizan hound',
            'Icelandic sheepdog',
            'Irish red and white setter',
            'Irish setter',
            'Irish terrier',
            'Irish water spaniel',
            'Irish wolfhound',
            'Italian greyhound',
            'Japanese chin',
            'Keeshond',
            'Kerry blue terrier',
            'Komondor',
            'Kuvasz',
            'Labrador retriever',
            'Lakeland terrier',
            'Leonberger',
            'Lhasa apso',
            'Lowchen',
            'Maltese',
            'Manchester terrier',
            'Mastiff',
            'Miniature schnauzer',
            'Neapolitan mastiff',
            'Newfoundland',
            'Norfolk terrier',
            'Norwegian buhund',
            'Norwegian elkhound',
            'Norwegian lundehund',
            'Norwich terrier',
            'Nova scotia duck tolling retriever',
            'Old english sheepdog',
            'Otterhound',
            'Papillon',
            'Parson russell terrier',
            'Pekingese',
            'Pembroke welsh corgi',
            'Petit basset griffon vendeen',
            'Pharaoh hound',
            'Plott',
            'Pointer',
            'Pomeranian',
            'Poodle',
            'Portuguese water dog',
            'Saint bernard',
            'Silky terrier',
            'Smooth fox terrier',
            'Tibetan mastiff',
            'Welsh springer spaniel',
            'Wirehaired pointing griffon',
            'Xoloitzcuintli',
            'Yorkshire terrier']


def face_detector(img):
    numpy_image = np.array(img)  
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def PILImage_to_tensor(img):
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img):
    # returns prediction vector for image located at img_path
    img = preprocess_input(PILImage_to_tensor(img))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img):
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151))


# load model
InceptionV3_model = Sequential()
InceptionV3_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
InceptionV3_model.add(Dropout(0.5))
InceptionV3_model.add(Dense(133, activation='sigmoid'))
InceptionV3_model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
InceptionV3_model.load_weights('save_models/weights.best.InceptionV3.hdf5')

def InceptionV3_predict_breed(img):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(PILImage_to_tensor(img))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def detect(img):
    is_human = face_detector(img)
    is_dog = dog_detector(img)
    if (is_human or is_dog):
        return InceptionV3_predict_breed(img)
    else:
        return 'Neither a dog nor human'


st.title("Dog Image Classification App")
st.write('\n')
img = Image.open('images/American_water_spaniel_00648.jpg')
show = st.image(img, use_column_width=True)

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file).resize((224, 224), Image.ANTIALIAS)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

# For newline
st.sidebar.write('\n')

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner('Classifying ...'):
            prediction = detect(u_img)
            time.sleep(2)
            st.success('Done!')

        st.sidebar.header("Algorithm Predicts: ")
        st.sidebar.write(prediction)
