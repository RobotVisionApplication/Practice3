import numpy as np

def compute_image_error(current_pts, target_pts):
    """current_pts, target_pts: Nx2 numpy arrays"""
    return (current_pts - target_pts).reshape(-1, 1)

def estimate_velocity_from_error(error, lam=0.1):
    """비례 제어 방식 (실제는 interaction matrix와 Jacobian 사용)"""
    return -lam * error.flatten()

def load_points(file):
    return np.load(file)

if __name__ == "__main__":
    current = load_points('current_corners.npy')
    target = load_points('target_corners.npy')

    error = compute_image_error(current, target)
    velocity_cmd = estimate_velocity_from_error(error)

    print("Image error:\n", error)
    print("Velocity command:\n", velocity_cmd)
