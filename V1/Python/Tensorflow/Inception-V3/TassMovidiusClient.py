############################################################################################
# Title: TASS Movidius Client
# Description: Client for testing sending images to TASS Movidius.
# Last Modified: 2018/03/07
############################################################################################

print("")
print("")
print("!! Welcome to TASS Movidius Client, please wait while the program initiates !!")
print("")

import os, sys

print("-- Running on Python "+sys.version)
print("")

import requests, json, cv2, time

print("-- Imported Required Modules")

class TassMovidiusClient():
    
    def __init__(self):

        self.addr = 'http://192.168.2.101:7455'
        self.apiUrl = self.addr + '/api/infer'
        self.positive = 'positive.png'
        self.negative = 'negative.png'
        self.content_type = 'image/jpeg'
        self.headers = {'content-type': self.content_type}

        print("-- TassMovidiusClient Initiated")
        
    def sendImage(self, image):
        
        img = cv2.imread(image)
        _, img_encoded = cv2.imencode('.png', img)
        response = requests.post(self.apiUrl, data=img_encoded.tostring(), headers=self.headers)
        
        print(json.loads(response.text))

TassMovidiusClient = TassMovidiusClient()
TassMovidiusClient.sendImage(TassMovidiusClient.positive)
time.sleep(5)
TassMovidiusClient.sendImage(TassMovidiusClient.negative)
time.sleep(5)