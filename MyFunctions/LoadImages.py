from os import listdir
from os.path import isfile, join

import numpy as np
import SimpleITK as sitk
import skimage.transform


class LoadImages:
    """
        Class to import CT or PET scan data. Class inputs are:

            ScanType: CT / PET.
            ScanClass: Image / Mask.
            ImgPath: Path to folder that contains the desired images.
            MskPath: Path to folder that contains the desired masks.
            n_Scans: The number of scans to import.
            CT_Max: Maximum HU threshold for CT scans.
            CT_Min: Minimum HU threshold for CT scans.
            ImgSize: The resized image size for all imported scans.
            ImgDepth: The resized image depth (number of images in a scan) for all imported scans.
            Orientation: The orientation in which scans are to imported.

        Packages required:
            from os import listdir
            from os.path import isfile, join
            import numpy as np
            import SimpleITK as sitk

    """

    def __init__(self, ScanType="None", ScanClass="None", ScanName="None", ImgPath="None", MskPath="None", n_Scans=0, CT_Max=3072, CT_Min=-1024, ImgSize=256, ImgDepth=256, Orientation="None", Orientations="None", FlipUD=False):
        self.ScanType = ScanType
        self.ScanClass = ScanClass
        self.ScanName = ScanName
        self.ImgPath = ImgPath
        self.MskPath = MskPath
        self.n_Scans = n_Scans
        self.CT_Max = CT_Max
        self.CT_Min = CT_Min
        self.ImgSize = ImgSize
        self.ImgDepth = ImgDepth
        self.Orientation = Orientation
        self.Orientations = Orientations
        self.FlipUD = FlipUD

    def LoadPredictions(self):

        files = [f for f in listdir(self.ImgPath)
                 if isfile(join(self.ImgPath, f))]
        print("Reading the following " +
              self.ScanType+" " + self.ScanClass + "s:")
        cnt = 0
        for file in files:
            if cnt == 0:
                SITK_IMG = sitk.ReadImage(
                    join(self.ImgPath, file).replace("\\", "/"))
                SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
                if self.FlipUD:
                    SITK_ARR = np.flipud(SITK_ARR)
                SITK_ARR = self.scanReorientation(SITK_ARR)
                SITK_ARR = self.MaskProcess(SITK_ARR)
                if self.ImgDepth != 0:
                    SITK_ARR = self.resizeResliceImage(IMG=SITK_ARR)
                else:
                    SITK_ARR = self.resizeImage(IMG=SITK_ARR)
                IMGS_LIST = SITK_ARR
                cnt += 1
                print(file)

            elif cnt < self.n_Scans:
                SITK_IMG = sitk.ReadImage(
                    join(self.ImgPath, file).replace("\\", "/"))
                SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
                if self.FlipUD:
                    SITK_ARR = np.flipud(SITK_ARR)
                SITK_ARR = self.scanReorientation(SITK_ARR)
                SITK_ARR = self.MaskProcess(SITK_ARR)
                if self.ImgDepth != 0:
                    SITK_ARR = self.resizeResliceImage(IMG=SITK_ARR)
                else:
                    SITK_ARR = self.resizeImage(IMG=SITK_ARR)
                IMGS_LIST = np.concatenate((IMGS_LIST, SITK_ARR))
                cnt += 1
                print(file)
            else:
                print("")
                break
        return IMGS_LIST

    def LoadScanEnsamble(self):
        OUT = np.zeros((256, 256, 256, 3))
        i = 0
        if self.ScanClass == "Image":
            SITK_IMG = sitk.ReadImage(
                join(self.ImgPath, self.ScanName).replace("\\", "/"))
        elif self.ScanClass == "Mask":
            SITK_IMG = sitk.ReadImage(
                join(self.MskPath, self.ScanName).replace("\\", "/"))
        SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
        METADATA = self.GetMetadata(SITK_IMG)
        ORIG_SIZE = SITK_ARR.shape
        if self.FlipUD:
            SITK_ARR = np.flipud(SITK_ARR)

        for ori in self.Orientations:
            self.Orientation = ori
            SITK_REOR = self.scanReorientation(SITK_ARR)
            if self.ScanClass == "Image" and self.ScanType == "CT":
                SITK_REOR = self.ctImageProcess(SITK_REOR)
            elif self.ScanClass == "Mask":
                SITK_ARR = self.MaskProcess(SITK_ARR)
            if self.ImgDepth == 0:
                SITK_REOR = self.resizeImage(IMG=SITK_REOR)
            else:
                SITK_REOR = self.resizeResliceImage(IMG=SITK_REOR)
            OUT[:, :, :, i] = SITK_REOR
            i = i + 1
        return OUT, ORIG_SIZE, METADATA

    def LoadScan(self):
        if self.ScanClass == "Image":
            SITK_IMG = sitk.ReadImage(
                join(self.ImgPath, self.ScanName).replace("\\", "/"))
        elif self.ScanClass == "Mask":
            SITK_IMG = sitk.ReadImage(
                join(self.MskPath, self.ScanName).replace("\\", "/"))
        SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
        METADATA = self.GetMetadata(SITK_IMG)
        ORIG_SIZE = SITK_ARR.shape
        if self.FlipUD:
            SITK_ARR = np.flipud(SITK_ARR)
        SITK_ARR = self.scanReorientation(SITK_ARR)
        if self.ScanClass == "Image" and self.ScanType == "CT":
            SITK_ARR = self.ctImageProcess(SITK_ARR)
        elif self.ScanClass == "Mask":
            SITK_ARR = self.MaskProcess(SITK_ARR)
        if self.ImgDepth == 0:
            if self.ImgSize == 0:
                return SITK_ARR, ORIG_SIZE, METADATA
            else:
                SITK_ARR = self.resizeImage(IMG=SITK_ARR)
        else:
            SITK_ARR = self.resizeResliceImage(IMG=SITK_ARR)
        return SITK_ARR, ORIG_SIZE, METADATA

    def LoadScans(self):
        """
            Function to load multple scans and resize them to be used to train the model. Orientation can be: Axial, Sagittal, Coronal
        """
        if self.ScanClass == "Image":
            files = [f for f in listdir(
                self.ImgPath) if isfile(join(self.ImgPath, f))]
        elif self.ScanClass == "Mask":
            files = [f for f in listdir(
                self.MskPath) if isfile(join(self.MskPath, f))]

        print("Reading the following " +
              self.ScanType+" " + self.ScanClass + "s:")
        cnt = 0
        for file in files:
            if cnt == 0:
                if self.ScanClass == "Image":
                    SITK_IMG = sitk.ReadImage(
                        join(self.ImgPath, file).replace("\\", "/"))
                elif self.ScanClass == "Mask":
                    SITK_IMG = sitk.ReadImage(
                        join(self.MskPath, file).replace("\\", "/"))
                SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
                if self.FlipUD:
                    SITK_ARR = np.flipud(SITK_ARR)
                SITK_ARR = self.scanReorientation(SITK_ARR)
                if self.ImgDepth != 0:
                    SITK_ARR = self.resizeResliceImage(IMG=SITK_ARR)
                else:
                    SITK_ARR = self.resizeImage(IMG=SITK_ARR)
                if self.ScanClass == "Image" and self.ScanType == "CT":
                    SITK_ARR = self.ctImageProcess(SITK_ARR)
                elif self.ScanClass == "Mask":
                    SITK_ARR = self.MaskProcess(SITK_ARR)
                IMGS_LIST = SITK_ARR
                cnt += 1
                print(file)
            elif cnt < self.n_Scans:
                if self.ScanClass == "Image":
                    SITK_IMG = sitk.ReadImage(
                        join(self.ImgPath, file).replace("\\", "/"))
                elif self.ScanClass == "Mask":
                    SITK_IMG = sitk.ReadImage(
                        join(self.MskPath, file).replace("\\", "/"))
                SITK_ARR = sitk.GetArrayFromImage(SITK_IMG)
                if self.FlipUD:
                    SITK_ARR = np.flipud(SITK_ARR)
                SITK_ARR = self.scanReorientation(SITK_ARR)
                if self.ImgDepth != 0:
                    SITK_ARR = self.resizeResliceImage(IMG=SITK_ARR)
                else:
                    SITK_ARR = self.resizeImage(IMG=SITK_ARR)
                if self.ScanClass == "Image" and self.ScanType == "CT":
                    SITK_ARR = self.ctImageProcess(SITK_ARR)
                elif self.ScanClass == "Mask":
                    SITK_ARR = self.MaskProcess(SITK_ARR)
                IMGS_LIST = np.concatenate((IMGS_LIST, SITK_ARR))
                cnt += 1
                print(file)
            else:
                print("")
                break
        return IMGS_LIST

    def resizeImage(self, IMG):
        if self.ScanClass == "Image":
            if self.ImgSize < IMG.shape[1]:
                resized_img = skimage.transform.resize(
                    image=IMG, output_shape=(IMG.shape[0], self.ImgSize, self.ImgSize))
            else:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    IMG.shape[0], self.ImgSize, self.ImgSize), anti_aliasing=True)
            return resized_img
        elif self.ScanClass == "Mask":
            if self.ImgSize < IMG.shape[1]:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    IMG.shape[0], self.ImgSize, self.ImgSize), preserve_range=True)
            else:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    IMG.shape[0], self.ImgSize, self.ImgSize), preserve_range=True)
            resized_img[resized_img >= 0.5] = 1
            resized_img[resized_img < 0.5] = 0
            resized_img = resized_img.astype('uint8')
            return resized_img

    def resizeResliceImage(self, IMG):
        if self.ScanClass == "Image":
            if self.ImgSize < IMG.shape[1]:
                resized_img = skimage.transform.resize(
                    image=IMG, output_shape=(self.ImgDepth, self.ImgSize, self.ImgSize))
            else:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    self.ImgDepth, self.ImgSize, self.ImgSize), anti_aliasing=True)
            return resized_img
        elif self.ScanClass == "Mask":
            if self.ImgSize < IMG.shape[1]:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    self.ImgDepth, self.ImgSize, self.ImgSize), preserve_range=True)
            else:
                resized_img = skimage.transform.resize(image=IMG, output_shape=(
                    self.ImgDepth, self.ImgSize, self.ImgSize), preserve_range=True)
            resized_img[resized_img >= 0.5] = 1
            resized_img[resized_img < 0.5] = 0
            resized_img = resized_img.astype('uint8')
            return resized_img

    def scanReorientation(self, DATA):
        if self.Orientation == "Axial":
            pass
        elif self.Orientation == "Sagittal":
            DATA = np.swapaxes(DATA, 1, 0)
            DATA = np.swapaxes(DATA, 2, 0)
        elif self.Orientation == "Coronal":
            DATA = np.swapaxes(DATA, 1, 0)
        return DATA

    def scanReorientationInverse(self, DATA):
        if self.Orientation == "Axial":
            pass
        elif self.Orientation == "Sagittal":
            DATA = np.swapaxes(DATA, 1, 0)
            DATA = np.swapaxes(DATA, 1, 2)
        elif self.Orientation == "Coronal":
            DATA = np.swapaxes(DATA, 1, 0)
        return DATA

    def GetMetadata(self, IMAGE):
        """Returns array METADATA = [SPACING,ORIGIN,DIRECTION,METADATA] """
        SPACING = IMAGE.GetSpacing()
        ORIGIN = IMAGE.GetOrigin()
        DIRECTION = IMAGE.GetDirection()
        METADATA = [SPACING, ORIGIN, DIRECTION]
        return METADATA

    def SetMetadata(self, IMAGE, METADATA):
        """Sets a given IMAGE with a given METADATA"""
        IMAGE.SetSpacing(METADATA[0])
        IMAGE.SetOrigin(METADATA[1])
        IMAGE.SetDirection(METADATA[2])

    def ctImageProcess(self, CTScan):
        """Function to set max and min HU values and then normalises the input to between 0 and 1."""
        CTScan = CTScan.astype(np.float32)
        CTScan[CTScan > self.CT_Max] = self.CT_Max
        CTScan[CTScan < self.CT_Min] = self.CT_Min
        CTScan += -np.min(CTScan)
        CTScan /= np.max(CTScan)
        return CTScan

    def MaskProcess(self, Mask):
        """Function to normalise the input between 0 and 1."""
        Mask = Mask.astype(np.float32)
        Mask += -np.min(Mask)
        Mask[Mask >= 0.8] = 1
        Mask[Mask < 0.8] = 0
        return Mask
