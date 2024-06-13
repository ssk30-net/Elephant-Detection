import torch
import numpy as np
import cv2
import time
from ultralytics.vit import RTDETR
import supervision as sv

class DETRClass:
    def __init__(self,capture_index):
        self.capture_index=capture_index
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using device:",self.device)
        self.model=RTDETR("rtdetr-l.pt")
        self.CLASS_NAMES_DICT=self.model.model.names
        print("Classes:",self.CLASS_NAMES_DICT)
        self.box_annotator=sv.BoxAnnotator(sv.ColorPalette.default(),thickness=3,text_thickness=3,text_scale=1.5)

    def plot_bbox(self,results,frame):
        # extract Results
        boxes=results[0].boxes.cpu().numpy()
        class_id=boxes.cls
        conf=boxes.conf
        xyxy=boxes.xyxy

        #plot the bounding boxes
        class_id=class_id.as_type(np.int32)
        detections=sv.Detections(xyxy=xyxy,class_id=class_id,confidence=conf)
        self.labels=[f"{self.CLASS_NAMES_DICT[class_id]}{confidence:.2f}" for xyxy,mask,confidence,class_id,track_id in detections]
        frame=self.box_annotator(frame,detections,self.labels)
        return frame
    def __call__(self):
        cap=cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        while cap.isOpened():
            start_time=time.perf_counter()
            ret,frame=cap.read()
            results=self.model.predict(frame)
            frame=self.plot_bbox(results,frame)
            end_time=time.pref_counter()
            fps=1/(end_time-start_time)
            cv2.putText(frame,f"FPS:{fps:2f}",(20,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("DETR_Object_Detction",frame)
            if cv2.waitKey(1)==ord("q"):
                break
        cap.release()
        cap.destroyAllWindows()

transformer_detector=DETRClass(0)
transformer_detector()