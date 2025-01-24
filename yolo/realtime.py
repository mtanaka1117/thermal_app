from ultralytics import YOLO
import cv2

# Initialize a YOLO-World model
model = YOLO("yolov5m_Objects365.pt")

# model.set_classes([""])

# ["Person", "Glasses", "Bottle", "Cup", "Handbag/Satchel", "Book",
#                     "Umbrella", "Watch", "Pen/Pencil", "Cell Phone",
#                     "Laptop", "Clock", "Keyboard", "Mouse", "Head Phone", "Remote",
#                     "Scissors", "Folder", "earphone", "Mask", "Tissue", "Wallet/Purse",
#                     "Tablet", "Key", "CD", "Stapler", "Eraser", "Lipstick"]

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラフレームの取得に失敗しました")
        break

    # YOLOで物体検出
    results = model.predict(source=frame, save=False, conf=0.5)

    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()

    # フレームを表示
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()