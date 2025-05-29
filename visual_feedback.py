import time
import numpy as np

# 로봇 제어 라이브러리 (예시용)
class DummyRobot:
    def __init__(self):
        self.joint_state = np.zeros(6)

    def movej(self, joints):
        print("Move to:", joints)
        self.joint_state = joints
        time.sleep(1)

    def speedl(self, dx, acc=0.1, t=0.1):
        print("Speed control:", dx)
        self.joint_state += dx * t
        time.sleep(t)

    def getj(self):
        return self.joint_state

def load_pose(path):
    return np.load(path)

def main_loop():
    robot = DummyRobot()
    ready_pose = load_pose("ready_pose.npy")

    robot.movej(ready_pose)

    for _ in range(10):  # 반복 제어
        # (실제 시스템에서는 camera에서 현재 코너 추출)
        current = np.load('current_corners.npy')
        target = np.load('target_corners.npy')

        from visual_feedback import compute_image_error, estimate_velocity_from_error
        err = compute_image_error(current, target)
        vel = estimate_velocity_from_error(err)

        robot.speedl(vel[:6])  # 6-DOF 제어만 사용했다고 가정

        if np.linalg.norm(err) < 1.0:
            print("Target reached.")
            break

if __name__ == "__main__":
    main_loop()