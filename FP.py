import cv2
import numpy as np
import cvzone
import speech_recognition as sr
import pyttsx3 as px
from twilio.rest import Client
import keys
import time

thres = 0.38  # Confidence threshold
#thres = 0.8
nmsThres = 0.1  # Non-maximum suppression threshold

# Load class names
classNames = []
classFile = 'names.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# Configuration files and paths
configPath = 'ssd_mobilenet.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#cap = cv2.VideoCapture('ELEPHANT/test_village.mp4')
cap = cv2.VideoCapture(0)

cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
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



while True:
    success, img = cap.read()

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    if len(classIds) > 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Ensure classId is within the valid range of classNames
            if 0 <= classId - 1 < len(classNames):
                # Check if the detected object is an elephant
                if classNames[classId - 1].lower() == 'elephant':
                    # Draw rectangle and display text for elephant
                    cvzone.cornerRect(img, box)
                    cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}%',
                                (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    send_sms_alert()  # Trigger SMS alert
                    alert_voice()  # Trigger voice alert

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
