### importing required libraries
import torch
import cv2
import time
from twilio.rest import Client
import speech_recognition as sr
import pyttsx3 as px

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)


    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):
    """
    --> This function taks results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        if n >=1:
            # Initialize Twilio client for sending messages
            account_sid = 'ACa241374d0887c64d6d6f39c1e1226e5f'
            auth_token = '51b7ccfd1a7324be9b5a695695a8700d'
            twilio_phone_number = '+13344633500'
            your_phone_number = '+919245536926'
            client = Client(account_sid, auth_token)
            message = client.messages.create(body="Elephant detected in the video!", from_=twilio_phone_number,to=your_phone_number)
            print("Elephant Detection")
            speakfromtext_cmd("Elephant detected")
        print("Elephant detected ")
        row = cord[i]
        if row[4] >= 0.85:  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBOx coordniates
            text_d = classes[int(labels[i])]

            if text_d == 'elephant':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  ## BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)  ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            elif text_d == 'noelephant':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  ## BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 0, 255), -1)  ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

    return frame


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    #model =  torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model = torch.hub.load('yolov5', 'custom',source='local', path='best.pt', force_reload=True)  ### The repo is stored locally

    classes = model.names  ### class names in string format

    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)  ### DETECTION HAPPENING HERE

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, classes=classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                print(f"[INFO] Exiting. . . ")
                cv2.imwrite("final_output.jpg", frame)  ## if you want to save he output result.

                break

    elif vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)

        if vid_out:  ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                frame_no += 1

        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()

        ## closing all windows
        cv2.destroyAllWindows()

### -------------------  speech alerts -------------------------------
# Initialize voice alerts
speech=sr.Recognizer()
# Sectioning
try:
    engine = px.init()  # Engine for text to speech
except ImportError:
        print('Requested Driver not found')
except RuntimeError:
        print('Driver Fails to Initialize')

# Section property of engine
voices = engine.getProperty('voices')

engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')  # setting the tts to zira the female voice
# for setting speech rate
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)

# Function for speaking the text given
def speakfromtext_cmd(cmd):
    engine.say(cmd)
    engine.runAndWait()
### -------------------  calling the main function-------------------------------

#main(vid_path="TIGER.mp4",vid_out="videoplayback.mp4") ### for custom video
main(vid_path=0,vid_out="test.mp4") #### for webcam

#main(img_path="PIC.jpeg")  ## for imagep