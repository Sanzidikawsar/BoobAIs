############################################################################################
# Title: TASS Movidius Server
# Description: Serves the TASS Movidius Classifier providing an API for interaction.
# Acknowledgements: Uses code from Intel movidius/ncsdk (https://github.com/movidius/ncsdk)
# Last Modified: 2018/03/07
############################################################################################

print("")
print("")
print("!! Welcome to TASS Movidius Server, please wait while the program initiates !!")
print("")

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("-- Running on Python "+sys.version)
print("")

import time,csv,getopt,json,time, cv2, jsonpickle

from flask import Flask, request, Response
from mvnc import mvncapi as mvnc
from tools.helpers import TassMovidiusHelpers
from datetime import datetime
from skimage.transform import resize

import numpy as np
import JumpWayMQTT.Device as JWMQTTdevice 

print("-- API Initiating ")
app = Flask(__name__) 
print("-- API Intiated ")
print("")

print("-- Imported Required Modules")

class TassMovidiusServer():
    
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
        print("-- TassMovidiusServer Initiated")
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
        
TassMovidiusServer = TassMovidiusServer()
TassMovidiusServer.loadInceptionRequirements()

# route http posts to this method
@app.route('/api/infer', methods=['POST'])
def test():
            
    humanStart = datetime.now()
    clockStart = time.time()
    
    print("-- INCEPTION V3 LIVE INFERENCE STARTING ")
    print("-- STARTED: : ", humanStart)
    print("")

    r = request
    nparr = np.fromstring(r.data, np.uint8)
    
    print("-- Loading Sample")
    fileName = str(clockStart)+'.png'
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(fileName,img)
    img = cv2.imread(fileName).astype(np.float32)
    print("-- Loaded Sample")
                    
    dx,dy,dz= img.shape
    delta=float(abs(dy-dx))
    
    if dx > dy: 
        
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
        
    else:
        
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
        
    img = cv2.resize(img, (TassMovidiusServer.reqsize, TassMovidiusServer.reqsize))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(3):
        
        img[:,:,i] = (img[:,:,i] - TassMovidiusServer.mean) * TassMovidiusServer.std

    detectionStart = datetime.now()
    detectionClockStart = time.time()
    
    print("-- DETECTION STARTING ")
    print("-- STARTED: : ", detectionStart)
    print("")

    TassMovidiusServer.graph.LoadTensor(img.astype(np.float16), 'user object')
    output, userobj = TassMovidiusServer.graph.GetResult()

    top_inds = output.argsort()[::-1][:5]

    detectionEnd = datetime.now()
    detectionClockEnd = time.time()

    identified = 0

    print("")
    print("-- DETECTION ENDING")
    print("-- ENDED: ", detectionEnd)
    print("-- TIME: {0}".format(detectionClockEnd - detectionClockStart))
    print("")
                    
    if output[top_inds[0]] > TassMovidiusServer._configs["ClassifierSettings"]["InceptionThreshold"] and TassMovidiusServer.categories[top_inds[0]] == "1":
        
        identified = identified + 1
        
        print("")
        print("TASS Identified IDC with a confidence of", str(output[top_inds[0]]))
        print("")

        TassMovidiusServer.jumpwayClient.publishToDeviceChannel(
                "Sensors",
                {
                    "Sensor":"CCTV",
                    "SensorID": TassMovidiusServer._configs["Cameras"][0]["ID"],
                    "SensorValue":"IDC: " + TassMovidiusServer.categories[top_inds[0]] + " (Confidence: " + str(output[top_inds[0]]) + ")"
                }
            )
            
        print("")

    #print(top_inds)
    #print(TassMovidiusServer.categories)

    print("".join(['*' for i in range(79)]))
    print('inception-v3 on NCS')
    print("".join(['*' for i in range(79)]))
    
    for i in range(2):
        
        print(top_inds[i], TassMovidiusServer.categories[top_inds[i]], output[top_inds[i]])

    print("".join(['*' for i in range(79)]))

    humanEnd = datetime.now()
    clockEnd = time.time()

    print("")
    print("-- INCEPTION V3 LIVE INFERENCE ENDING")
    print("-- ENDED: ", humanEnd)
    print("-- TESTED: ", 1)
    print("-- IDENTIFIED: ", identified)
    print("-- TIME(secs): {0}".format(clockEnd - clockStart))
    print("")

    if identified:
        
        message = "IDC Detected!"

    else:
        
        message = "IDC Not Detected!"
        
    response = {
        'Response': 'OK', 
        'Results': 1, 
        'ResponseMessage': message
    }
    
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")
    
app.run(host="192.168.2.101", port=7455)