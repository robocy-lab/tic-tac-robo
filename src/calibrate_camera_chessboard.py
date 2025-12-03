import cv2
import numpy as np
import glob
import os
import subprocess

def capture_calibration_images(num_images=30, calib_dir="calibration_images"):
    """Capture images for camera calibration using a chessboard."""
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    
    # Check if rpicam-jpeg is available
    try:
        subprocess.run(["rpicam-jpeg", "--help"], capture_output=True, check=True)
        use_rpicam = True
        print("Using rpicam-jpeg for image capture")
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_rpicam = False
        print("rpicam-jpeg not found, using OpenCV camera")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
    
    print("Prepare the chessboard and press Enter to start capturing...")
    input()
    
    for i in range(num_images):
        print(f"Capturing image {i+1}/{num_images}. Move the board to a new position and press Enter...")
        input()
        output_path = f"{calib_dir}/calib_{i+1:02d}.jpg"
        
        if use_rpicam:
            subprocess.run(["rpicam-jpeg", "--output", output_path, "--timeout", "100", "--vflip"], check=True)
        else:
            ret, frame = cap.read()
            if ret:
                # Flip vertically to match rpicam behavior
                frame = cv2.flip(frame, 0)
                cv2.imwrite(output_path, frame)
            else:
                print(f"Failed to capture image {i+1}")
                continue
        
        print(f"Saved: {output_path}")
    
    if not use_rpicam:
        cap.release()
    
    print(f"Capture complete! Now run the calibration.")

def calibrate_camera_chessboard(calib_dir="calibration_images", checkerboard=(9, 6), square_size_mm=None):
    """Calibrate the camera using a chessboard."""
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    if square_size_mm is not None:
        objp *= square_size_mm / 1000.0  # in meters
    objpoints = []
    imgpoints = []
    images = glob.glob(f'{calib_dir}/*.jpg')
    if len(images) < 10:
        print(f"Only {len(images)} images found. At least 15-20 are recommended.")
    print(f"Processing {len(images)} images...")
    successful = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            successful += 1
            img_c = img.copy()
            cv2.drawChessboardCorners(img_c, checkerboard, corners2, ret)
            cv2.imwrite(f"{calib_dir}/detected_{os.path.basename(fname)}", img_c)
        else:
            print(f"Failed to find corners in {fname}")
    print(f"Successfully processed {successful} out of {len(images)} images")
    if successful < 10:
        print("Not enough successful detections for good calibration!")
        return
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None,
        flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
    )
    print(f"\n=== CALIBRATION RESULTS ===")
    print(f"RMS error: {ret:.6f}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print(f"Mean reprojection error: {mean_error:.6f} pixels")
    np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms_error=ret, mean_error=mean_error)
    print(f"\n=== CODE TO INSERT INTO april_tag_lib.py ===")
    print(f"CAMERA_MATRIX = np.array({camera_matrix.tolist()})")
    print(f"DIST_COEFFS = np.array({dist_coeffs.flatten().tolist()})")
    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    print("Select mode:\n1 — Capture images\n2 — Calibration\n")
    mode = input("Enter 1 or 2: ").strip()
    if mode == "1":
        capture_calibration_images()
    elif mode == "2":
        # Specify the square size in mm if you know it (e.g., 25)
        calibrate_camera_chessboard()
    else:
        print("Unknown mode!")
