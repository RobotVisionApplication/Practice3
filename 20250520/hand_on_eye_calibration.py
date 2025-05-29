import numpy as np
import cv2
import os
import glob

def load_matrix_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 빈 줄 제거

    print(f"\n[DEBUG] Reading file: {file_path}")
    for i, line in enumerate(lines):
        tokens = line.strip().split()
        print(f"Line {i+1} ({len(tokens)} tokens): {tokens}")
    
    if len(lines) != 4:
        raise ValueError(f"[FORMAT ERROR] {file_path}: Expected 4 lines, got {len(lines)}")

    matrix = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"[FORMAT ERROR] {file_path}: Line {i+1} has {len(parts)} elements (expected 4)")
        matrix.append([float(x) for x in parts])

    return np.array(matrix)

def pose_to_rt(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    return R, t

def run_hand_eye_from_dataset(dataset_path):
    camera_files = sorted(glob.glob(os.path.join(dataset_path, "camera_*.txt")))
    hand_files   = sorted(glob.glob(os.path.join(dataset_path, "hand_*.txt")))

    if len(camera_files) != len(hand_files):
        raise ValueError("Not Matching Pairs")

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    for cam_path, hand_path in zip(camera_files, hand_files):
        T_cam = load_matrix_from_txt(cam_path)      
        T_hand = load_matrix_from_txt(hand_path)  

        T_target2cam = np.linalg.inv(T_cam)      
        R_tc, t_tc = pose_to_rt(T_target2cam)
        R_gb, t_gb = pose_to_rt(T_hand)

        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(t_tc)
        R_gripper2base_list.append(R_gb)
        t_gripper2base_list.append(t_gb)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    print("Calibration Done:")
    print("Rotation:\n", R_cam2gripper)
    print("Translation:\n", t_cam2gripper)
    print("Camera to Gripper Transform:\n", T_cam2gripper)

    return T_cam2gripper

if __name__ == "__main__":
    dataset_path = "/home/sh/robot_vision_application/20250520/data2"
    run_hand_eye_from_dataset(dataset_path)