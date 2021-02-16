class CreatePaths:

    def __init__(self, DeviceFlag, ScanTypeFlag, TrainTestFlag, PET_Type = "Heart"):

        """ 
            DeviceFlag   = "PC" for PC, DeviceFlag   = "Laptop" for Laptop
            ScanTypeFlag = "CT" for CT, ScanTypeFlag = "PET" for PET   
            TrainTestFlag = "Train" for Train, "Test"
        """ 

        self.Device = DeviceFlag
        self.TrainTest = TrainTestFlag
        self.ScanType = ScanTypeFlag
        self.data_path = self.dataPath()

    def dataPath(self):
        if self.Device == "PC":
            DATA_PATH = "D:/Masters_Repo"
        elif self.Device == "Laptop": 
            DATA_PATH = "C:/Users/JANDRE/Documents/DataRepository/"
        else:
            print("\nError: Device Flag input should be PC or Laptop.\n")

        if self.TrainTest == "Train":
            DATA_PATH = DATA_PATH + "/TrainingData"
        elif self.ScanType == "Test": 
            DATA_PATH = DATA_PATH + "/TestingData"
        else:
            print("\nError: TrainTest Flag input should be Train or Test.\n")

        if self.ScanType == "CT":
            DATA_PATH = DATA_PATH + "/CT"
        elif self.ScanType == "PET": 
            DATA_PATH = DATA_PATH + "/PET"
        else:
            print("\nError: Scan Type Flag input should be CT or PET.\n")
        return DATA_PATH

    def imgPath(self):
        IMG_PATH = self.data_path + "/imgs"
        IMG_PATH = IMG_PATH.replace("\\","/")
        return IMG_PATH

    def mskPath(self):
        MSK_PATH = self.data_path + "/masks"
        MSK_PATH = MSK_PATH.replace("\\","/")
        return MSK_PATH

    def outputPath(self):       
        if self.Device == "PC":
            DATA_PATH = "D:/Output"
        elif self.Device == "Laptop": 
            DATA_PATH = _
        else:
            print("\nError: Device Flag input should be PC or Laptop.\n")

        if self.ScanType == "CT":
            DATA_PATH + "/CT"
        elif self.ScanType == "PET": 
            DATA_PATH + "/PET"
        else:
            print("\nError: Scan Type Flag input should be CT or PET.\n")
        return DATA_PATH
    