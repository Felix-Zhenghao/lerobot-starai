# Before running this script, you need to ensure the camera index is correct.
# The index may vary based on your system configuration.
# You can check like this `v4l2-ctl --device=/dev/video0 --list-formats-ext``
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    print("摄像头已打开")
else:
    print("无法打开摄像头")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Online Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()    
cv2.destroyAllWindows()