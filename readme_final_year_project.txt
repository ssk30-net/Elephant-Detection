Readme File 
Final Year project models 
Keys file for sending message alerts via Twilio to the receiver this is only for SSD and EfficientDet models 
Libraries Used:
FP.py-SSD model:
1.opencv-4.5.5.64
2.numpy-1.26.0
3.cvzone-1.6.1
4.pyttsx3-2.90
5.twilio-8.10.0
6.Speech_recognition-3.10.0
EfficientDet model-efficiendet.py:
1.opencv-4.5.5.64
2.numpy-1.26.0
3.cvzone-1.6.1
4.pyttsx3-2.90
5.twilio-8.10.0
6.Speech_recognition-3.10.0
Customized Yolo v5 model:
1.supervision-0.21.0
2.ultralytics-8.2.29
3.opencv-4.5.5.64
4.numpy-1.26.0
Detr model:
1.supervision-0.21.0
2.ultralytics-8.2.29
3.opencv-4.5.5.64
4.numpy-1.26.0
5.torch- 1.11.0

SSD Model
 video input in cv2.VideoCapture(0)#for live footage 
				cv2.VideoCapture(vido path )#for video
				
EfficientDet Model
 video input in cv2.VideoCapture(0)#for live footage 
				cv2.VideoCapture(vido path )#for video	
Yolo v5 Model 
	cv2.VideoCapture(0 )#for live footage
Detr Model:
	transformer_detector=DETRClass(0) # for camera feed and real time
	transformer_detector=DETRClass(Video path) #for video input 
	