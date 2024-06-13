import cv2 as cv
import numpy as np
import speech_recognition as sr
import pyttsx3 as px
from twilio.rest import Client
import keys
import time

cap = cv.VideoCapture(0)
#cap = cv.VideoCapture('ELEPHANT/test_village.mp4')
whT = 320
confThreshold = 0.38
nmsThreshold = 0.8

#### LOAD MODEL
## Coco Names
classesFile = "names.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = [line.strip() for line in f.readlines()]

## Model Files
modelConfiguration = "effi_ele.cfg"
modelWeights = "effi_ele.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# For voice alert
speech = sr.Recognizer()

# Sectioning
try:
    engine = px.init()  # Engine for text to speech
except ImportError:
    print('Requested Driver not found')
except RuntimeError:
    print('Driver Fails to Initialize')

# Section property of engine
voices = engine.getProperty('voices')

engine.setProperty('voice',
                   'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')  # setting the tts to zira the female voice
# for setting speech rate
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)


# Function for speaking the text given
def speak_from_text_cmd(cmd):
    engine.say(cmd)
    engine.runAndWait()


# For sms Alert
client = Client(keys.account_sid, keys.auth_token)
voice_alert_count = 0
sms_alert_sent = False


# Function to send SMS alert
def send_sms_alert():
    global sms_alert_sent
    if not sms_alert_sent:
        client.messages.create(
            to=keys.target_number,
            from_=keys.twilio_number,
            body="Elephant detected in the area!"
        )
        sms_alert_sent = True


# Function to alert using voice
def alert_voice():
    global voice_alert_count
    for i in range(5):
        print("Voice alert: Elephant detected in the area!")
        speak_from_text_cmd("Elephant detected in the area!")
        voice_alert_count += 1




# Function to find objects in the frame
# Function to find objects in the frame
def find_objects(outputs, img):
    elephant_detected = False
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == classNames.index('elephant'):
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                elephant_detected = True

    if elephant_detected:
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = indices[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                       (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            send_sms_alert()  # Trigger SMS alert
            alert_voice()  # Trigger voice alert


while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    find_objects(outputs, img)

    cv.imshow('Test', img)

    cv.waitKey(1)
