import os
import numpy as np
import skimage.transform
import SimpleITK as sitk


class RunModels:
    """
        Class to ease running a single, or multiple models. 

        runModel: Run a single model, that model being provided to this class as an input, Model. (1 model)

        runModels: Run an array of models, passed in an array to the Model input parameter. (Usually 3 models)

        runEnsemble: Run an extra model, EnsModel, on the output of runModels (Usually 4 models)
    """

    def __init__(self, OutPath, ScanName, Scan, Scan_Size, Scan_Metadata, Model, Orientation, Threshold=0.5, Batch_Size=8, Scans="", Orientations="", EnsModel="", EnsWeights=[1, 1, 1]):
        self.OutPath = OutPath
        self.ScanName = ScanName
        self.Scan = Scan
        self.Scan_Size = Scan_Size
        self.Scan_Metadata = Scan_Metadata
        self.Model = Model
        self.Batch_Size = Batch_Size
        self.Orientation = Orientation
        self.Threshold = Threshold
        self.Scans = Scans
        self.Orientations = Orientations
        self.EnsModel = EnsModel
        self.EnsWeights = EnsWeights

    def SetMetadata(self, IMAGE):
        """Sets a given IMAGE with a given METADATA"""
        IMAGE.SetSpacing(self.Scan_Metadata[0])
        IMAGE.SetOrigin(self.Scan_Metadata[1])
        IMAGE.SetDirection(self.Scan_Metadata[2])

    def scanReorientationInverse(self, DATA):
        if self.Orientation == "Axial":
            pass
        elif self.Orientation == "Sagittal":
            DATA = np.swapaxes(DATA, 1, 0)
            DATA = np.swapaxes(DATA, 1, 2)
        elif self.Orientation == "Coronal":
            DATA = np.swapaxes(DATA, 1, 0)
        return DATA

    def runModel(self):
        self.Scan = np.expand_dims(self.Scan, axis=3)
        Model_Out = self.Model.predict(self.Scan, self.Batch_Size, verbose=1)
        Model_Out = np.squeeze(Model_Out)
        Model_Out = self.scanReorientationInverse(Model_Out)
        Model_Out = skimage.transform.resize(
            Model_Out, (self.Scan_Size[0], self.Scan_Size[1], self.Scan_Size[1]))
        Model_Out[Model_Out > self.Threshold] = 1
        Model_Out[Model_Out < self.Threshold] = 0
        Model_Out[Model_Out == self.Threshold] = 1
        Model_Out = sitk.GetImageFromArray(Model_Out)
        self.SetMetadata(Model_Out)
        WritePath = os.path.join(
            self.OutPath, self.ScanName+"_"+self.Orientation+".nii.gz").replace("\\", "/")
        sitk.WriteImage(Model_Out, WritePath)
        return print("Done! \t Path: " + WritePath)

    def runModels(self, Limit=False, ReturnArray=False, Resize=True, Verbose=1):
        if Resize:
            OUT = np.zeros(
                (self.Scan_Size[0], self.Scan_Size[1], self.Scan_Size[1]))
        else:
            OUT = np.zeros((256, 256, 256))

        i = 0
        for model in self.Model:
            scan = self.Scans[:, :, :, i]
            scan = np.expand_dims(scan, axis=3)
            Temp_Out = model.predict(scan, self.Batch_Size, verbose=Verbose)
            Temp_Out = np.squeeze(Temp_Out)
            self.Orientation = self.Orientations[i]
            Temp_Out = self.scanReorientationInverse(Temp_Out)
            if Resize:
                Temp_Out = skimage.transform.resize(
                    Temp_Out, (self.Scan_Size[0], self.Scan_Size[1], self.Scan_Size[1]))
            OUT = OUT + Temp_Out*self.EnsWeights[i]
            i = i+1
        OUT = OUT / 3
        if Limit:
            OUT[OUT > self.Threshold] = 1
            OUT[OUT < self.Threshold] = 0
            OUT[OUT == self.Threshold] = 1
        if ReturnArray:
            #print("Returning Array..")
            return OUT
        else:
            OUT = sitk.GetImageFromArray(OUT)
            self.SetMetadata(OUT)
            OutName = self.ScanName.replace("CT", "lungs")
            WritePath = os.path.join(
                self.OutPath, OutName+".nii.gz").replace("\\", "/")
            sitk.WriteImage(OUT, WritePath)
            return print("Done! \t Path: " + WritePath)

    def runEnsemble(self, Limit=False, ReturnArray=False, Resize=True, Verbose=1):

        OUT = np.zeros((256, 256, 256))
        i = 0

        for model in self.Model:
            scan = self.Scans[:, :, :, i]
            scan = np.expand_dims(scan, axis=3)
            Temp_Out = model.predict(scan, self.Batch_Size, verbose=Verbose)
            Temp_Out = np.squeeze(Temp_Out)
            self.Orientation = self.Orientations[i]
            Temp_Out = self.scanReorientationInverse(Temp_Out)
            OUT = OUT + Temp_Out
            i = i+1

        OUT = np.expand_dims(OUT, axis=3)
        Ensemble = self.EnsModel.predict(OUT, self.Batch_Size, verbose=Verbose)
        Ensemble = np.squeeze(Ensemble)
        if Resize:
            Ensemble = skimage.transform.resize(
                Ensemble, (self.Scan_Size[0], self.Scan_Size[1], self.Scan_Size[1]))

        if Limit:
            Ensemble[Ensemble > self.Threshold] = 1
            Ensemble[Ensemble < self.Threshold] = 0
            Ensemble[Ensemble == self.Threshold] = 1
        if ReturnArray:
            #print("Returning Array..")
            return Ensemble
        else:
            Ensemble = sitk.GetImageFromArray(Ensemble)
            self.SetMetadata(Ensemble)
            WritePath = os.path.join(
                self.OutPath, self.ScanName+".nii.gz").replace("\\", "/")
            sitk.WriteImage(Ensemble, WritePath)
            return print("Done! \t Path: " + WritePath)
