import cv2
from ultralytics import YOLO
import cvzone

# Load model and get class names
model = YOLO('yolo12s.pt')
names = model.names

# Video source
cap = cv2.VideoCapture("falldown.mp4")


frame_count=0

# Mouse callback (for debugging pixel position)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True,classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), 1, 1)
            h=y2-y1
            w=x2-x1
            thresh=h-w
#            print(thresh) 
            if 'person' in name:
                if thresh <0:
                   cvzone.putTextRect(frame,f'{"person_fall"}',(x1,y2 +12),1,1)
                   cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

          
    # Show frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
