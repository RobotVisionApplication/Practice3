import cv2
import numpy as np
import os

class CameraViewer:
    def __init__(self, save_path='saved_corners.txt', checkerboard_size=(3, 2)):
        self.save_path = save_path
        self.enable_detector = False
        self.corners = []
        self.cap = cv2.VideoCapture(0)
        self.checkerboard_size = checkerboard_size  # (cols, rows), 내부 코너 수

    def detect_corners_checkerboard(self, gray, vis_img):
        found, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found and len(corners) == self.checkerboard_size[0] * self.checkerboard_size[1]:
            # refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            self.corners = [tuple(pt.ravel()) for pt in corners]
            # draw on image
            cv2.drawChessboardCorners(vis_img, self.checkerboard_size, corners, found)
        else:
            self.corners = []

        return self.corners

    def save_corners_to_txt(self):
        expected = self.checkerboard_size[0] * self.checkerboard_size[1]
        if len(self.corners) != expected:
            print(f"[ERROR] Expected {expected} corners, but got {len(self.corners)}. Not saving.")
            return
        with open(self.save_path, 'w') as f:
            for pt in self.corners:
                f.write(f"{pt[0]} {pt[1]}\n")
        print(f"[INFO] Saved {len(self.corners)} corners to {self.save_path}")

    def run(self):
        print("[c] checkerboard detect | [s] save | [n] stop | [ESC] exit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            vis_frame = frame.copy()
            if self.enable_detector:
                self.detect_corners_checkerboard(gray, vis_frame)

            # draw circles manually if needed
            for pt in self.corners:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            cv2.imshow("Camera View", vis_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                self.enable_detector = True
                print(f"[INFO] Checkerboard detector enabled. Size = {self.checkerboard_size}")
            elif key == ord('n'):
                self.enable_detector = False
                print("[INFO] Detector disabled.")
            elif key == ord('s'):
                self.save_corners_to_txt()
            elif key == 27:  # ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = CameraViewer(checkerboard_size=(3, 4))  # 내부 코너 수 (cols, rows)
    viewer.run()
