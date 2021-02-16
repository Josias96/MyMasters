#Function to ease the use of certain fixed paths across multiple devices


def dataPath(FLAG):
    
    """Flag = 1 for PC, Flag = 2 for Laptop""" 

    if FLAG == 0:
        DATA_PATH = "D:/Masters_Repo/Diseased"
    elif FLAG == 1: 
        DATA_PATH = "C:/Users/JANDRE/Documents/DataRepository/CB_named_scans"
    else:
        print("\nError: Input should be 0 or 1.\n")
    return DATA_PATH
    
def outputPath(FLAG):

    """Flag = 1 for PC, Flag = 2 for Laptop""" 
       
    if FLAG == 0:
        OUTPUT_PATH = "D:/Masters_Repo/MyPredictions/StruisBaaiModel"
    elif FLAG == 1:
        OUTPUT_PATH = "C:/Users/JANDRE/Documents/DataRepository/Outputs"
    else:
        print("\nError: Input should be 0 or 1.\n")
    return OUTPUT_PATH

def imgPath(DATA_PATH):
    IMG_PATH = DATA_PATH + "/"+"imgs"
    IMG_PATH = IMG_PATH.replace("\\","/")
    return IMG_PATH

def mskPath(DATA_PATH):
    MSK_PATH = DATA_PATH +"/"+"masks"
    MSK_PATH = MSK_PATH.replace("\\","/")
    return MSK_PATH

