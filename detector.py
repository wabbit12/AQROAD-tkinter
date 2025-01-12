import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue

class RoadSignDetector:
    def __init__(self):

        self.engine = None
        self.speaking = False
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
        self.sign_classes = {
            0: {
                "name": "Speed limit (20km/h)",
                "description": "Maximum speed limit of 20 kilometers per hour. Typically found in highly pedestrianized areas or zones requiring extra caution."
            },
            1: {
                "name": "Speed limit (30km/h)",
                "description": "Maximum speed limit of 30 kilometers per hour. Common in residential areas and school zones."
            },
            2: {
                "name": "Speed limit (50km/h)",
                "description": "Maximum speed limit of 50 kilometers per hour. Standard speed limit in urban areas."
            },
            3: {
                "name": "Speed limit (60km/h)",
                "description": "Maximum speed limit of 60 kilometers per hour. Often found on major urban roads."
            },
            4: {
                "name": "Speed limit (70km/h)",
                "description": "Maximum speed limit of 70 kilometers per hour. Typical for roads transitioning between urban and rural areas."
            },
            5: {
                "name": "Speed limit (80km/h)",
                "description": "Maximum speed limit of 80 kilometers per hour. Common on rural roads and highways."
            },
            6: {
                "name": "End of speed limit (80km/h)",
                "description": "Indicates the end of the 80km/h speed limit zone. Return to standard speed limits."
            },
            7: {
                "name": "Speed limit (100km/h)",
                "description": "Maximum speed limit of 100 kilometers per hour. Typically found on highways and motorways."
            },
            8: {
                "name": "Speed limit (120km/h)",
                "description": "Maximum speed limit of 120 kilometers per hour. Common on major highways and motorways."
            },
            9: {
                "name": "No passing",
                "description": "Overtaking or passing other vehicles is prohibited. Stay in your lane."
            },
            10: {
                "name": "No passing for vehicles over 3.5 metric tons",
                "description": "Heavy vehicles weighing more than 3.5 metric tons are not allowed to overtake other vehicles."
            },
            11: {
                "name": "Right-of-way at the next intersection",
                "description": "You have priority at the upcoming intersection. Other vehicles must yield to you."
            },
            12: {
                "name": "Priority road",
                "description": "You are on a priority road. You have right of way at intersections."
            },
            13: {
                "name": "Yield",
                "description": "You must give way to other traffic. Stop if necessary and proceed only when safe."
            },
            14: {
                "name": "Stop",
                "description": "Come to a complete stop. Proceed only when safe and after yielding to other traffic."
            },
            15: {
                "name": "No vehicles",
                "description": "No vehicles of any kind are permitted beyond this point."
            },
            16: {
                "name": "Vehicles over 3.5 metric tons prohibited",
                "description": "Heavy vehicles exceeding 3.5 metric tons are not allowed on this road."
            },
            17: {
                "name": "No entry",
                "description": "Entry forbidden for all vehicles. Do not enter."
            },
            18: {
                "name": "General caution",
                "description": "Warning for a general hazard ahead. Proceed with increased attention."
            },
            19: {
                "name": "Dangerous curve to the left",
                "description": "Sharp bend ahead to the left. Reduce speed and prepare to turn."
            },
            20: {
                "name": "Dangerous curve to the right",
                "description": "Sharp bend ahead to the right. Reduce speed and prepare to turn."
            },
            21: {
                "name": "Double curve",
                "description": "Series of bends ahead. First curve could be either left or right. Reduce speed."
            },
            22: {
                "name": "Bumpy road",
                "description": "Road surface is uneven ahead. Reduce speed and prepare for bumps."
            },
            23: {
                "name": "Slippery road",
                "description": "Road surface may be slippery. Reduce speed and increase following distance."
            },
            24: {
                "name": "Road narrows on the right",
                "description": "The road becomes narrower on the right side ahead. Adjust position accordingly."
            },
            25: {
                "name": "Road work",
                "description": "Construction or maintenance work ahead. Reduce speed and watch for workers."
            },
            26: {
                "name": "Traffic signals",
                "description": "Traffic light ahead. Prepare to stop if the signal is red."
            },
            27: {
                "name": "Pedestrians",
                "description": "Pedestrian crossing ahead. Watch for people crossing the road."
            },
            28: {
                "name": "Children crossing",
                "description": "School zone or playground area. Watch for children crossing the road."
            },
            29: {
                "name": "Bicycles crossing",
                "description": "Bicycle crossing ahead. Watch for cyclists crossing or joining the road."
            },
            30: {
                "name": "Beware of ice/snow",
                "description": "Risk of ice or snow on the road. Adjust driving style for winter conditions."
            },
            31: {
                "name": "Wild animals crossing",
                "description": "Wildlife crossing area ahead. Watch for animals on the road."
            },
            32: {
                "name": "End of all speed and passing limits",
                "description": "Previous speed and passing restrictions end. Standard traffic rules apply."
            },
            33: {
                "name": "Turn right ahead",
                "description": "Mandatory right turn ahead. Prepare to turn right at the intersection."
            },
            34: {
                "name": "Turn left ahead",
                "description": "Mandatory left turn ahead. Prepare to turn left at the intersection."
            },
            35: {
                "name": "Ahead only",
                "description": "Must proceed straight ahead. No turning allowed."
            },
            36: {
                "name": "Go straight or right",
                "description": "Must either continue straight or turn right. No left turn allowed."
            },
            37: {
                "name": "Go straight or left",
                "description": "Must either continue straight or turn left. No right turn allowed."
            },
            38: {
                "name": "Keep right",
                "description": "Stay on the right side of the road or obstacle ahead."
            },
            39: {
                "name": "Keep left",
                "description": "Stay on the left side of the road or obstacle ahead."
            },
            40: {
                "name": "Roundabout mandatory",
                "description": "Must enter and follow the roundabout in the indicated direction."
            },
            41: {
                "name": "End of no passing",
                "description": "End of no-overtaking zone. Passing other vehicles is now allowed."
            },
            42: {
                "name": "End of no passing by vehicles over 3.5 metric tons",
                "description": "End of no-overtaking zone for heavy vehicles. Trucks may now pass other vehicles."
            }
        }
        
        self.model = YOLO('models/best.pt')
        self.conf_threshold = 0.5

    def _speech_worker(self):
        self.engine = pyttsx3.init()
        
        while True:
            try:
                text = self.speech_queue.get()
                if text is None:
                    break
                
                self.speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.speaking = False
                
            except Exception as e:
                print(f"Speech error: {e}")
                self.speaking = False
            finally:
                self.speech_queue.task_done()

    def speak_description(self, text):
        try:
            self.speech_queue.put_nowait(text)
        except queue.Full:
            print("Speech queue is full, skipping this announcement")
        
    def detect_signs(self, frame):
        results = self.model(frame)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence > self.conf_threshold:
                    sign_info = self.sign_classes.get(class_id, {
                        "name": "Unknown Sign",
                        "description": "Sign not recognized"
                    })
                    detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'name': sign_info["name"],
                        'description': sign_info["description"],
                        'confidence': confidence
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):

        frame_copy = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            label = f"{det['name']} ({det['confidence']:.2f})"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy

    def __del__(self):
        if hasattr(self, 'speech_queue'):
            self.speech_queue.put(None)
            if hasattr(self, 'speech_thread'):
                self.speech_thread.join(timeout=1)