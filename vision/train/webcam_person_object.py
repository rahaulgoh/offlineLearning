import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPlainTextEdit,
    QHBoxLayout, QVBoxLayout
)

CONF_TH = 0.4

class YoloViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Viewer (Video + Log)")
        self.resize(1100, 650)

        self.count_history = deque(maxlen=9)

        # UI Widgets
        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumWidth(720)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumWidth(350)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Detection Log:"))
        right_panel.addWidget(self.log)

        layout = QHBoxLayout()
        layout.addWidget(self.video_label, stretch = 3)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        layout.addWidget(right_widget, stretch = 2)

        self.setLayout(layout)


        # Video capture
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # state
        self.last_person_state = False
        self.last_log_time = 0.0

        # timer loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

        self._log("UI started.")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")
    
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self._log("Camera read failed.")
            return
        
        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Inference
        results = self.model(frame, imgsz=640, conf=CONF_TH, verbose=False)[0]

        person_present = False
        person_count = 0
        color = (0, 255, 0)

        # Draw boxes + decide person is present
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = self.model.names.get(cls_id, "")

            x1, y1, x2, y2 = map(int, b.xyxy[0])

            label = f"{name.capitalize()} {conf:.2f}"
            if name == "person":
                person_count += 1
                self.count_history.append(person_count)

                sorted_counts = sorted(self.count_history)
                stable_count = sorted_counts[len(sorted_counts)//2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            person_present = person_count > 0
            now = time.time()

            # Frame change logging
            if (
                person_present != self.last_person_state or 
                person_count != getattr(self, "last_person_count", -1)
            ) and (now - self.last_log_time) > 0.5:
                self._log(f"People detected: {stable_count}")
                self.last_person_state = person_present
                self.last_person_count = person_count
                self.last_log_time = now

            # Convert frame (BGR -> RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self._log("Shutting down.")
        event.accept()

def main():
    app = QApplication(sys.argv)
    viewer = YoloViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()