############################################################################################
# Title: TASS Movidius Classifier
# Description: Test classification of local testing images.
# Acknowledgements: Uses code from Intel movidius/ncsdk (https://github.com/movidius/ncsdk)
# Last Modified: 2018/03/06
############################################################################################

############################################################################################
#
#    CLASSIFIER MODE:
#    
#       Classifier & IoT JumpWay configuration can be found in data/confs.json
#
#    Commandline Arguments:
#    
#        - InceptionTest: This mode sets the program to classify testing images using Inception V3
#
#    Example Usage:
#
#        $ python3.5 TassMovidiusClassifier.py InceptionTest
#
############################################################################################
          
print("")
print("")
print("!! Welcome to TASS Movidius Classifier, please wait while the program initiates !!")
print("")

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("-- Running on Python "+sys.version)
print("")

from mvnc import mvncapi as mvnc
import sys,os,time,csv,getopt,json,time 
import cv2
import numpy as np
import JumpWayMQTT.Device as JWMQTTdevice 

from tools.helpers import TassMovidiusHelpers
from datetime import datetime
from skimage.transform import resize

print("-- Imported Required Modules")

class TassMovidiusClassifier():
    
    def __init__(self):

        self._configs = {}
        self.movidius = None
        self.jumpwayClient = None
        self.cameraStream = None
        self.imagePath = None
        
        self.mean = 128
        self.std = 1/128
        
        self.categories = []
        self.graphfile = None
        self.graph = None
        self.reqsize = None

        self.extensions = [
            ".jpg",
            ".png"
        ]
        
        self.CheckDevices()
        self.TassMovidiusHelpers = TassMovidiusHelpers()
        self._configs = self.TassMovidiusHelpers.loadConfigs()
        self.startMQTT()
        
        print("")
        print("-- TassMovidiusClassifier Initiated")
        print("")
            
    def CheckDevices(self):
        
        #mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('!! WARNING! No Movidius Devices Found !!')
            quit()

        self.movidius = mvnc.Device(devices[0])
        self.movidius.OpenDevice()
        
        print("-- Movidius Connected")
            
    def allocateGraph(self,graphfile):

        self.graph = self.movidius.AllocateGraph(graphfile)
        
    def loadInceptionRequirements(self):
        
        self.reqsize = self._configs["ClassifierSettings"]["image_size"]
        
        with open(self._configs["ClassifierSettings"]["NetworkPath"] + self._configs["ClassifierSettings"]["InceptionGraph"], mode='rb') as f:
            
            self.graphfile = f.read()

        self.allocateGraph(self.graphfile)
        
        print("-- Allocated Graph OK")
            
        with open(self._configs["ClassifierSettings"]["NetworkPath"] + 'model/classes.txt', 'r') as f:
            
            for line in f:
                
                cat = line.split('\n')[0]
                
                if cat != 'classes':
                    
                    self.categories.append(cat)
                    
            f.close()
        
        print("-- Categories Loaded OK:", len(self.categories))
            
    def startMQTT(self):

        try:

            self.jumpwayClient = JWMQTTdevice.DeviceConnection({
                "locationID": self._configs["IoTJumpWaySettings"]["SystemLocation"],
                "zoneID": self._configs["IoTJumpWaySettings"]["SystemZone"],
                "deviceId": self._configs["IoTJumpWaySettings"]["SystemDeviceID"],
                "deviceName": self._configs["IoTJumpWaySettings"]["SystemDeviceName"],
                "username": self._configs["IoTJumpWayMQTTSettings"]["MQTTUsername"],
                "password": self._configs["IoTJumpWayMQTTSettings"]["MQTTPassword"]
            })

        except Exception as e:
            print(str(e))
            sys.exit()

        self.jumpwayClient.connectToDevice()
        
        print("-- IoT JumpWay Initiated")
        
TassMovidiusClassifier = TassMovidiusClassifier()
	
def main(argv):
    
    try:
    
        if argv[0] == "InceptionTest":
            
            humanStart = datetime.now()
            clockStart = time.time()
            
            print("-- INCEPTION V3 TEST MODE STARTING ")
            print("-- STARTED: : ", humanStart)
            print("")
                
            TassMovidiusClassifier.loadInceptionRequirements()
                        
            rootdir= TassMovidiusClassifier._configs["ClassifierSettings"]["NetworkPath"] + TassMovidiusClassifier._configs["ClassifierSettings"]["InceptionImagePath"]
                
            files = 0
            identified = 0

            for file in os.listdir(rootdir):
                
                if file.endswith('.jpg','.jpeg','.png','.gif'):
                    
                    files = files + 1
                    fileName = rootdir+file
            
                    print("")
                    print("-- Loaded Test Image", fileName)
                    img = cv2.imread(fileName).astype(np.float32)
                    print("")
                    
                    dx,dy,dz= img.shape
                    delta=float(abs(dy-dx))
                    
                    if dx > dy: 
                        
                        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
                        
                    else:
                        
                        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
                        
                    img = cv2.resize(img, (TassMovidiusClassifier.reqsize, TassMovidiusClassifier.reqsize))
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    for i in range(3):
                        
                        img[:,:,i] = (img[:,:,i] - TassMovidiusClassifier.mean) * TassMovidiusClassifier.std
        
                    detectionStart = datetime.now()
                    detectionClockStart = time.time()
                    
                    print("-- DETECTION STARTING ")
                    print("-- STARTED: : ", detectionStart)
                    print("")

                    TassMovidiusClassifier.graph.LoadTensor(img.astype(np.float16), 'user object')
                    output, userobj = TassMovidiusClassifier.graph.GetResult()

                    top_inds = output.argsort()[::-1][:5]
        
                    detectionEnd = datetime.now()
                    detectionClockEnd = time.time()

                    print("")
                    print("-- DETECTION ENDING")
                    print("-- ENDED: ", detectionEnd)
                    print("-- TIME: {0}".format(detectionClockEnd - detectionClockStart))
                    print("")
                    
                    if output[top_inds[0]] > TassMovidiusClassifier._configs["ClassifierSettings"]["InceptionThreshold"]:
                        
                        identified = identified + 1
                        
                        print("")
                        print("TASS Identified ", TassMovidiusClassifier.categories[top_inds[0]], "With A Confidence Of", str(output[top_inds[0]]))
                        print("")

                        TassMovidiusClassifier.jumpwayClient.publishToDeviceChannel(
                                "Sensors",
                                {
                                    "Sensor":"CCTV",
                                    "SensorID": TassMovidiusClassifier._configs["Cameras"][0]["ID"],
                                    "SensorValue":"OBJECT: " + TassMovidiusClassifier.categories[top_inds[0]] + " (Confidence: " + str(output[top_inds[0]]) + ")"
                                }
                            )
                            
                        print("")

                    #print(top_inds)
                    #print(TassMovidiusClassifier.categories)

                    print("".join(['*' for i in range(79)]))
                    print('inception-v3 on NCS')
                    print("".join(['*' for i in range(79)]))
                    
                    for i in range(2):
                        
                        print(top_inds[i], TassMovidiusClassifier.categories[top_inds[i]], output[top_inds[i]])

                    print("".join(['*' for i in range(79)]))
        
            humanEnd = datetime.now()
            clockEnd = time.time()

            print("")
            print("-- INCEPTION V3 TEST MODE ENDING")
            print("-- ENDED: ", humanEnd)
            print("-- TIME(secs): {0}".format(clockEnd - clockStart))
            print("")
            
            TassMovidiusClassifier.graph.DeallocateGraph()
            TassMovidiusClassifier.movidius.CloseDevice()
            sys.exit()
                            
            print("!! SHUTTING DOWN !!")
            print("")
            
        else:
            
            print("**ERROR** Check Your Commandline Arguments")
            print("")
        
    except:
        
        print("**ERROR** Check Your Commandline Arguments")
        print("")

if __name__ == "__main__":
    	
	main(sys.argv[1:])