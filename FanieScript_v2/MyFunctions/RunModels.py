import os
import numpy as np
import skimage.transform
import SimpleITK as sitk

class RunModels:
    """

    """
    
    def __init__(self,OutPath, ScanName, Scan, Scan_Size, Scan_Metadata, Model, Orientation, Threshold = 0.5, Batch_Size = 8, Scans = "", Orientations = ""):
        self.OutPath = OutPath
        self.ScanName = ScanName
        self.Scan = Scan
        self.Scan_Size = Scan_Size
        self.Scan_Metadata = Scan_Metadata
        self.Model = Model
        self.Batch_Size = Batch_Size
        self.Orientation =  Orientation
        self.Threshold = Threshold
        self.Scans = Scans
        self.Orientations = Orientations
    def SetMetadata(self,IMAGE):
        """Sets a given IMAGE with a given METADATA"""
        IMAGE.SetSpacing(self.Scan_Metadata[0])
        IMAGE.SetOrigin(self.Scan_Metadata[1])
        IMAGE.SetDirection(self.Scan_Metadata[2])
    
    def scanReorientationInverse(self,DATA):
        if self.Orientation == "Axial":
            pass
        elif self.Orientation == "Sagittal":
            DATA = np.swapaxes(DATA,1,0)
            DATA = np.swapaxes(DATA,1,2)
        elif self.Orientation == "Coronal":
            DATA = np.swapaxes(DATA,1,0)
        return DATA

    def runModel(self):
        self.Scan = np.expand_dims(self.Scan,axis=3)
        Model_Out = self.Model.predict(self.Scan,self.Batch_Size,verbose=1)
        Model_Out = np.squeeze(Model_Out)
        Model_Out = self.scanReorientationInverse(Model_Out)
        Model_Out = skimage.transform.resize(Model_Out,(self.Scan_Size[0],self.Scan_Size[1],self.Scan_Size[1]))
        Model_Out[Model_Out>self.Threshold]  = 1
        Model_Out[Model_Out<self.Threshold]  = 0
        Model_Out[Model_Out==self.Threshold] = 1
        Model_Out = sitk.GetImageFromArray(Model_Out)
        self.SetMetadata(Model_Out)
        WritePath = os.path.join(self.OutPath,self.ScanName+"_"+self.Orientation+".nii.gz").replace("\\","/")
        sitk.WriteImage(Model_Out,WritePath)
        return print("Done! \t Path: " + WritePath)

    def runModels(self, Limit=False):
        OUT = np.zeros((self.Scan_Size[0],self.Scan_Size[1],self.Scan_Size[1]))
        i = 0
        for model in self.Model:
            scan = self.Scans[:,:,:,i]
            scan = np.expand_dims(scan,axis=3)
            Temp_Out = model.predict(scan,self.Batch_Size,verbose=1)
            Temp_Out = np.squeeze(Temp_Out)
            self.Orientation = self.Orientations[i]
            Temp_Out = self.scanReorientationInverse(Temp_Out)
            Temp_Out = skimage.transform.resize(Temp_Out,(self.Scan_Size[0],self.Scan_Size[1],self.Scan_Size[1]))
            OUT = OUT + Temp_Out
            i = i+1
        OUT = OUT / 3 
        if Limit:
                OUT[OUT>self.Threshold]  = 1
                OUT[OUT<self.Threshold]  = 0
                OUT[OUT==self.Threshold] = 1
        OUT = sitk.GetImageFromArray(OUT)
        self.SetMetadata(OUT)
        WritePath = os.path.join(self.OutPath,self.ScanName+".nii.gz").replace("\\","/")
        sitk.WriteImage(OUT,WritePath)
        return print("Done! \t Path: " + WritePath)