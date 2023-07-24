from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture('ppe-1.mp4')

model = YOLO('ppe.pt')

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2, = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            #confidence
            conf = math.floor(box.conf * 100)
            #class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 50:
                if currentClass == "Hardhat" or currentClass == "Mask" or currentClass == "Safety Vest":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif currentClass == "NO-Hardhat" or currentClass == "NO-Mask" or currentClass == "NO-Safety Vest":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)