#Functions to load and show CT scans
import numpy as np
import os
from os import listdir
from os.path import isfile, join

import SimpleITK as sitk 
import skimage
from skimage.util import montage as montage2d
from skimage import transform
import matplotlib.pyplot as plt
import pickle


def ctImageProcess(CTScan, CT_MAX, CT_MIN):

    """Function to set max and min HU values and then normalises to between 0 and 1."""
    CTScan[CTScan > CT_MAX] = CT_MAX
    CTScan[CTScan < CT_MIN] = CT_MIN
    CTScan = normalise(CTScan)
    return CTScan

def normalise(a):
    a = a.astype(np.float32)
    a = a - np.min(a)
    a = a / np.max(a)
    return a


def PETImageProcess(PET_Scan):

    """Function to normalise PET scan."""
    PET_Scan = normalise(PET_Scan)
    return PET_Scan

def ctLoadScans(IMG_PATH,MAX_IMAGE_NUM,CT_MAX,CT_MIN,IMAGE_SIZE,MASK_FLAG,ORIENTATION):

    """Function to load multple CT scans and resize them to be used to train the model. 
    
       Orientation can be: RL_AP, AP_SI, RL_SI"""

    files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH,f))]
    cnt = 0
    if MASK_FLAG == 0:
        print("Reading the following CT scans:")
    else:
        print("Reading the following CT masks:")
    for file in files:
        if cnt == 0:
            SITK_IMG  = sitk.ReadImage(os.path.join(IMG_PATH,file))
            SITK_ARR  = sitk.GetArrayFromImage(SITK_IMG)
            SITK_ARR = SITK_ARR.astype(np.float32)
            if ORIENTATION == "RL_AP":
                pass
            elif ORIENTATION == "AP_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
                SITK_ARR = np.swapaxes(SITK_ARR,2,0)
            elif ORIENTATION == "RL_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
            ORIG_SIZE = SITK_ARR.shape
            if MASK_FLAG == 0:
                SITK_ARR  = ctImageProcess(SITK_ARR,CT_MAX,CT_MIN)
            else:
                SITK_ARR = normalise(SITK_ARR)
            SITK_ARR  = resizeResliceImage(SITK_ARR,IMAGE_SIZE,IMAGE_SIZE) 
            if MASK_FLAG == 1:
                #Note to self, after interpolation values change from 1. This resets those values to 1.
                SITK_ARR[SITK_ARR > 0.8] = 1
                SITK_ARR[SITK_ARR == 0.8] = 1
                SITK_ARR[SITK_ARR < 0.8] = 0
            
            IMGS_LIST = SITK_ARR
            cnt += 1
            print(file)
        elif cnt == MAX_IMAGE_NUM:
            print("\n")
            break
        else:
            SITK_IMG  = sitk.ReadImage(os.path.join(IMG_PATH, file))
            SITK_ARR  = sitk.GetArrayFromImage(SITK_IMG)
            SITK_ARR = SITK_ARR.astype(np.float32)
            if ORIENTATION == "RL_AP":
                pass        
            elif ORIENTATION == "AP_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
                SITK_ARR = np.swapaxes(SITK_ARR,2,0)
            elif ORIENTATION == "RL_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
            if MASK_FLAG == 0:
                SITK_ARR  = ctImageProcess(SITK_ARR,CT_MAX,CT_MIN)
            else:
                SITK_ARR = normalise(SITK_ARR)
            SITK_ARR  = resizeResliceImage(SITK_ARR,IMAGE_SIZE,IMAGE_SIZE) 
            if MASK_FLAG == 1: 
                SITK_ARR[SITK_ARR > 0.8] = 1
                SITK_ARR[SITK_ARR == 0.8] = 1
                SITK_ARR[SITK_ARR < 0.8] = 0
            IMGS_LIST = np.concatenate((IMGS_LIST, SITK_ARR))
            cnt += 1
            print(file)
    return IMGS_LIST, ORIG_SIZE

def PETLoadScans(IMG_PATH,MAX_IMAGE_NUM,IMAGE_SIZE,MASK_FLAG,ORIENTATION):

    """Function to load multple PET scans and resize them to be used to train the model. 
    
       Orientation can be: RL_AP, AP_SI, RL_SI"""

    files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH,f))]
    cnt = 0
    if MASK_FLAG == 0:
        print("Reading the following PET scans:")
    else:
        print("Reading the following PET masks:")
    for file in files:
        if cnt == 0:
            SITK_IMG  = sitk.ReadImage(os.path.join(IMG_PATH,file))
            SITK_ARR  = sitk.GetArrayFromImage(SITK_IMG)
            SITK_ARR = SITK_ARR.astype(np.float32)
            if ORIENTATION == "RL_AP":
                pass
            elif ORIENTATION == "AP_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
                SITK_ARR = np.swapaxes(SITK_ARR,2,0)
            elif ORIENTATION == "RL_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
            ORIG_SIZE = SITK_ARR.shape
            if MASK_FLAG == 0:
                SITK_ARR  = PETImageProcess(SITK_ARR)
            SITK_ARR  = resizeResliceImage(SITK_ARR,IMAGE_SIZE,IMAGE_SIZE) 
            if MASK_FLAG == 1:
                #Note to self, after interpolation values change from 1. This resets those values to 1.
                SITK_ARR[SITK_ARR > 0.8] = 1
                SITK_ARR[SITK_ARR == 0.8] = 1
                SITK_ARR[SITK_ARR < 0.8] = 0
            IMGS_LIST = SITK_ARR
            cnt += 1
            print(file)
        elif cnt == MAX_IMAGE_NUM:
            print("\n")
            break
        else:
            SITK_IMG  = sitk.ReadImage(os.path.join(IMG_PATH, file))
            SITK_ARR  = sitk.GetArrayFromImage(SITK_IMG)
            SITK_ARR = SITK_ARR.astype(np.float32)
            if ORIENTATION == "RL_AP":
                pass        
            elif ORIENTATION == "AP_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
                SITK_ARR = np.swapaxes(SITK_ARR,2,0)
            elif ORIENTATION == "RL_SI":
                SITK_ARR = np.swapaxes(SITK_ARR,1,0)
            if MASK_FLAG == 0:
                SITK_ARR  = PETImageProcess(SITK_ARR)
            SITK_ARR  = resizeResliceImage(SITK_ARR,IMAGE_SIZE,IMAGE_SIZE) 
            if MASK_FLAG == 1: 
                SITK_ARR[SITK_ARR > 0.8] = 1
                SITK_ARR[SITK_ARR == 0.8] = 1
                SITK_ARR[SITK_ARR < 0.8] = 0
            IMGS_LIST = np.concatenate((IMGS_LIST, SITK_ARR))
            cnt += 1
            print(file)
    return IMGS_LIST, ORIG_SIZE

def showCTMontage(IMG,SIZE):
    plt.figure(figsize=(SIZE, SIZE))
    plt.imshow(montage2d(IMG),alpha=1, cmap='gray')
    plt.axis('off')
    plt.show()

def showCTImage(IMG,SIZE):
    plt.figure(figsize=(SIZE,SIZE))
    plt.imshow(IMG,alpha=1,cmap='gray')
    plt.axis('off')
    plt.show()

def resizeImage(IMG,IMAGE_SIZE):

    """Function that uses skimage to resize CT's."""

    RESCALED_IMAGE = skimage.transform.resize(IMG,[IMG.shape[0],IMAGE_SIZE,IMAGE_SIZE])
    return RESCALED_IMAGE

def resizeResliceImage(IMG,IMAGE_SIZE,SLICE_NUMBER):
    resized_img = skimage.transform.resize(IMG,(SLICE_NUMBER,IMAGE_SIZE,IMAGE_SIZE))
    return resized_img

def ctLoadScan(DATA_PATH,SCAN_NAME,IMAGE_SIZE,IMAGE_SLICES,CT_MAX,CT_MIN):
    """Set IMAGE_SLICES = 0 if you do not wish to reslice"""
    SITK_IMG  = sitk.ReadImage(os.path.join(DATA_PATH,SCAN_NAME).replace("\\","/"))
    METADATA = GetMetadata(SITK_IMG)
    SITK_ARR  = sitk.GetArrayFromImage(SITK_IMG)
    SITK_ARR  = ctImageProcess(SITK_ARR,CT_MAX,CT_MIN)
    ORIG_SIZE = SITK_ARR.shape
    if IMAGE_SLICES == 0:
        SITK_ARR  = resizeImage(SITK_ARR,IMAGE_SIZE)
    else:
        SITK_ARR  = resizeResliceImage(SITK_ARR,IMAGE_SIZE,IMAGE_SIZE)
    return SITK_ARR,ORIG_SIZE,METADATA

def scanOrientation(DATA, ORIENTATION):
    if ORIENTATION == "RL_AP":
        pass
    elif ORIENTATION == "AP_SI":
        DATA = np.swapaxes(DATA,1,0)
        DATA = np.swapaxes(DATA,2,0)
    elif ORIENTATION == "RL_SI":
        DATA = np.swapaxes(DATA,1,0)
    return DATA

def scanOrientationInverse(DATA,ORIENTATION):
    if ORIENTATION == "RL_AP":
        pass
    elif ORIENTATION == "AP_SI":
        DATA = np.swapaxes(DATA,1,0)
        DATA = np.swapaxes(DATA,1,2)
    elif ORIENTATION == "RL_SI":
        DATA = np.swapaxes(DATA,1,0)
    return DATA

def GetMetadata(IMAGE):
    """Returns array METADATA = [SPACING,ORIGIN,DIRECTION,METADATA] """
    SPACING   = IMAGE.GetSpacing()
    ORIGIN    = IMAGE.GetOrigin()
    DIRECTION = IMAGE.GetDirection()
    METADATA = [SPACING,ORIGIN,DIRECTION]
    return METADATA

def SetMetadata(IMAGE,METADATA):
    """Sets a given IMAGE with a given METADATA"""
    IMAGE.SetSpacing(METADATA[0])
    IMAGE.SetOrigin(METADATA[1])
    IMAGE.SetDirection(METADATA[2])
    
    
def runModel(IMAGE_PATH,OUTPUT_PATH,SCAN_NAME,MODEL,BATCH_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN,ORIENTATION,MAX,VERSION):
    SCAN,ORIG_SIZE,METADATA = ctLoadScan(IMAGE_PATH,SCAN_NAME,IMAGE_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN)
    SCAN = scanOrientation(SCAN,ORIENTATION)
    SCAN = np.expand_dims(SCAN,axis=3)
    MODEL_OUT = MODEL.predict(SCAN,BATCH_SIZE,verbose=1)
    MODEL_OUT_IMAGE = np.squeeze(MODEL_OUT)
    MODEL_OUT_IMAGE = scanOrientationInverse(MODEL_OUT_IMAGE,ORIENTATION)
    MODEL_OUT_RESIZED = skimage.transform.resize(MODEL_OUT_IMAGE,(ORIG_SIZE[0],ORIG_SIZE[1],ORIG_SIZE[1]))
    MODEL_OUT_RESIZED[MODEL_OUT_RESIZED>MAX]  = 1
    MODEL_OUT_RESIZED[MODEL_OUT_RESIZED<MAX]  = 0
    MODEL_OUT_RESIZED[MODEL_OUT_RESIZED==MAX] = 1
    MODEL_OUT_IMAGE = sitk.GetImageFromArray(MODEL_OUT_RESIZED)
    SetMetadata(MODEL_OUT_IMAGE,METADATA)
    WRITE_PATH = os.path.join(OUTPUT_PATH,"Predict_"+ORIENTATION+"_"+VERSION+"_"+SCAN_NAME).replace("\\","/")
    sitk.WriteImage(MODEL_OUT_IMAGE,WRITE_PATH)
    return print("Done!")

def runModels(IMAGE_PATH,OUTPUT_PATH,MODEL_NAME,SCAN_NAME,MODELS,BATCH_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN,ORIENTATION_ENSAMBLE,MAX):
    SCAN_ORIG,ORIG_SIZE,METADATA = ctLoadScan(IMAGE_PATH,SCAN_NAME,IMAGE_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN)
    ENSAMBLE = np.empty(ORIG_SIZE)
    cnt = 0
    for model in MODELS:
        SCAN = scanOrientation(SCAN_ORIG,ORIENTATION_ENSAMBLE[cnt])
        SCAN = np.expand_dims(SCAN,axis=3)
        MODEL_OUT = model.predict(SCAN,BATCH_SIZE,verbose=1)
        MODEL_OUT_IMAGE = np.squeeze(MODEL_OUT)
        MODEL_OUT_IMAGE = scanOrientationInverse(MODEL_OUT_IMAGE,ORIENTATION_ENSAMBLE[cnt])
        MODEL_OUT_RESIZED = skimage.transform.resize(MODEL_OUT_IMAGE,(ORIG_SIZE[0],ORIG_SIZE[1],ORIG_SIZE[1]))
        ENSAMBLE = ENSAMBLE + MODEL_OUT_RESIZED
        cnt += 1
    ENSAMBLE = ENSAMBLE/np.size(MODELS)
    ENSAMBLE[ENSAMBLE>MAX]  = 1
    ENSAMBLE[ENSAMBLE<MAX]  = 0
    ENSAMBLE[ENSAMBLE==MAX] = 1
    ENSAMBLE_OUT_IMAGE = sitk.GetImageFromArray(ENSAMBLE)
    SetMetadata(ENSAMBLE_OUT_IMAGE,METADATA)
    WRITE_PATH = os.path.join(OUTPUT_PATH,MODEL_NAME+"_"+SCAN_NAME).replace("\\","/")
    sitk.WriteImage(ENSAMBLE_OUT_IMAGE,WRITE_PATH)
    return print("Done!")

def runModels2_toets(IMAGE_PATH,OUTPUT_PATH,MODEL_NAME,SCAN_NAME,MODELS,BATCH_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN,ORIENTATION_ENSAMBLE,MAX,VERSION):
    SCAN_ORIG,ORIG_SIZE,METADATA = ctLoadScan(IMAGE_PATH,SCAN_NAME,IMAGE_SIZE,IMAGE_SIZE,CT_MAX,CT_MIN)
    ENSAMBLE = np.empty(ORIG_SIZE)
    Temp = np.empty([256,256,256])
    cnt = 0
    for model in MODELS:
        SCAN = scanOrientation(SCAN_ORIG,ORIENTATION_ENSAMBLE[cnt])
        SCAN = np.expand_dims(SCAN,axis=3)
        MODEL_OUT = model.predict(SCAN,BATCH_SIZE,verbose=1)
        MODEL_OUT_IMAGE = np.squeeze(MODEL_OUT)
        MODEL_OUT_IMAGE = scanOrientationInverse(MODEL_OUT_IMAGE,ORIENTATION_ENSAMBLE[cnt])
        Temp = Temp + MODEL_OUT_IMAGE
        cnt += 1
    Temp = Temp/np.size(MODELS)
    Temp[Temp>MAX]  = 1
    Temp[Temp<MAX]  = 0
    Temp[Temp==MAX] = 1
    ENSAMBLE = skimage.transform.resize(Temp,(ORIG_SIZE[0],ORIG_SIZE[1],ORIG_SIZE[1]))
    ENSAMBLE = np.rint(ENSAMBLE)
    ENSAMBLE_OUT_IMAGE = sitk.GetImageFromArray(ENSAMBLE)
    SetMetadata(ENSAMBLE_OUT_IMAGE,METADATA)
    WRITE_PATH = os.path.join(OUTPUT_PATH,MODEL_NAME +"_"+VERSION+"_"+SCAN_NAME).replace("\\","/")
    sitk.WriteImage(ENSAMBLE_OUT_IMAGE,WRITE_PATH)
    return print("Done!")


def storePickle(DICT_STRING,history):
    with open(DICT_STRING, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return print("Done!")

def readPickle(DICT_STRING):
    with open(DICT_STRING, 'rb') as file_pi:
        history =  pickle.load(file_pi)
        return history
    
def plotHistory(dict_string,grid_on = True,range=0):
    history = readPickle(dict_string)
    plt.figure(figsize=(15, 5))
    if range != 0:
        plt.plot(history['accuracy'][0:range])
        plt.plot(history['val_accuracy'][0:range])
    else:
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])

    plt.title('Model Accuracy: ' + dict_string)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'], loc='upper left')
    if grid_on:
        plt.grid(b=True)
    plt.show()

    plt.figure(figsize=(15, 5))
    if range != 0:
        plt.plot(history['loss'][0:range])
        plt.plot(history['val_loss'][0:range])
    else:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

    plt.title('Model loss: ' + dict_string)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'], loc='upper left')
    if grid_on:
        plt.grid(b=True)
    plt.show()

def loadScans(IMGS_PATH,MSKS_PATH,num_images,IMAGE_SIZE,ORIENTATION,CT=False, PET=False,CT_MAX=0,CT_MIN=0):

    """Set either CT or PET as true"""
    if CT:
    # -------------------CT-----------------------
        SCAN_IMGS, _ = ctLoadScans(
            IMGS_PATH, num_images, CT_MAX, CT_MIN, IMAGE_SIZE, 0, ORIENTATION)
        SCAN_MASKS, ORIG_SIZE = ctLoadScans(
            MSKS_PATH, num_images, CT_MAX, CT_MIN, IMAGE_SIZE, 1, ORIENTATION)
    # --------------------------------------------
    if PET:
    # -------------------PET----------------------
        SCAN_IMGS, _ = PETLoadScans(
            IMGS_PATH, num_images, IMAGE_SIZE, 0, ORIENTATION)
        SCAN_MASKS, ORIG_SIZE = PETLoadScans(
            MSKS_PATH, num_images, IMAGE_SIZE, 1, ORIENTATION)
        # --------------------------------------------
    return SCAN_IMGS, SCAN_MASKS, ORIG_SIZE

        