#Functions to create geometric shapes in CT Scans for the purpose of 3D shape completion

import numpy as np
import cv2 as cv 

def createSphere(ARRAY,IMAGE_SIZE):

    """Create sphere function, bigger circle when subtracting then when adding."""

    maskType = np.random.randint(low=0, high=2,size=1)
    if maskType == 1:
        radius = np.max(np.random.randint(low=5, high=21, size=1))
    else: 
        radius = np.max(np.random.randint(low=5, high=36, size=1))
    centrePoint = np.random.randint(low=radius, high=IMAGE_SIZE-radius, size=2)
    highSlice = ARRAY.shape[0]
    highSlice = highSlice-radius
    sliceStart = np.max(np.random.randint(low=radius, high=highSlice, size=1))
    arrayCopy = np.zeros(ARRAY.shape)
    for cnt in range(radius):
        if cnt>0:
            sliceOfArrayUp   = arrayCopy[sliceStart+cnt,:,:]
            sliceOfArrayDown = arrayCopy[sliceStart-cnt,:,:]
            cv.circle(sliceOfArrayUp, (centrePoint[0],centrePoint[1]), radius-cnt, cv.COLOR_BGR2GRAY, -1)
            cv.circle(sliceOfArrayDown, (centrePoint[0],centrePoint[1]), radius-cnt, cv.COLOR_BGR2GRAY, -1)
            arrayCopy[sliceStart+cnt,:,:] = sliceOfArrayUp
            arrayCopy[sliceStart-cnt,:,:] = sliceOfArrayDown
        else:
            sliceOfArray = arrayCopy[sliceStart,:,:]
            cv.circle(sliceOfArray,(centrePoint[0],centrePoint[1]), radius, cv.COLOR_BGR2GRAY, -1)
            arrayCopy[sliceStart,:,:] = sliceOfArray
    print("Slice start:",sliceStart,"\tRadius:",radius,"\tValue:",maskType[0])
    if maskType == 1:
        X_sphere = ARRAY+arrayCopy
    else:
        X_sphere = ARRAY-arrayCopy
    X_sphere[X_sphere>1] = 1
    X_sphere[X_sphere<0] = 0
    print("Done")
    return sliceStart,radius,X_sphere,maskType


def augmentWithSphere(ARRAY,ITT,IMAGE_SIZE):
    for cnt in range(ITT):
        if cnt == 0:
            sliceStart,radius,X_sphere,maskType = createSphere(ARRAY,IMAGE_SIZE)
        else:
            sliceStart,radius,X_sphere,maskType = createSphere(X_sphere,IMAGE_SIZE)
            print(cnt)
    return(X_sphere)

#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox2D(img):
    if np.max(img) == 0:
        print("No mask in slice..")
        rmin = -1
        rmax = -1 
        cmin = -1
        cmax = -1
    else:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def bbox3D(imgs):
    cnt = 0
    flag_first_slice = 0
    flag_last_slice = 0
    for img in imgs:
        if np.max(img) == 0:
            rmin = -1
            rmax = -1 
            cmin = -1
            cmax = -1
            cnt = cnt+1
        else:
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            if flag_first_slice == 0:
                rmin_img = rmin
                rmax_img = rmax
                cmin_img = cmin
                cmax_img = cmax
                flag_first_slice = cnt
                cnt=cnt+1
            else:
                if rmin_img > rmin:
                    rmin_img = rmin
                if rmax_img < rmax:
                    rmax_img = rmax
                if cmin_img > cmin:
                    cmin_img = cmin
                if cmax_img < cmax:
                    cmax_img = cmax
                flag_last_slice = cnt
                cnt = cnt+1
    return rmin_img, rmax_img, cmin_img, cmax_img, flag_first_slice, flag_last_slice

def boundmask(image):
    rmin_img, rmax_img, cmin_img, cmax_img, flag_first_slice, flag_last_slice = bbox3D(image)
    output = image[flag_first_slice:flag_last_slice,rmin_img:rmax_img,cmin_img:cmax_img]
    return output