import time
import json
import logging
from typing import Dict, List, Tuple
import functools
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

# -----------------------------
# Logging setup
# -----------------------------
class SessionLogger:
    def __init__(self, base_logs_dir: str = "logs"):
        self.base_logs_dir = Path(base_logs_dir)
        self.session_dir = self._create_session_directory()
        self.logger = self._setup_logger()
        self.image_counter = 0
        
        self.logger.info(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Session directory: {self.session_dir}")
    
    def _create_session_directory(self) -> Path:
        """Create a new session directory with incremental numbering"""
        self.base_logs_dir.mkdir(exist_ok=True)
        
        # Find the next session number
        existing_sessions = [d for d in self.base_logs_dir.iterdir() if d.is_dir() and d.name.startswith('session_')]
        session_numbers = []
        for session in existing_sessions:
            try:
                num = int(session.name.split('_')[1])
                session_numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        next_session_num = max(session_numbers, default=0) + 1
        session_dir = self.base_logs_dir / f"session_{next_session_num:03d}"
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "images").mkdir(exist_ok=True)
        (session_dir / "debug_images").mkdir(exist_ok=True)
        (session_dir / "data").mkdir(exist_ok=True)
        
        return session_dir
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(f"aruco_session_{self.session_dir.name}")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.session_dir / "session.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%H:%M:%S.%f'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def save_image(self, image: np.ndarray, name: str, subdir: str = "images") -> Path:
        """Save image with incremental naming"""
        self.image_counter += 1
        filename = f"{self.image_counter:04d}_{name}"
        if not filename.endswith(('.jpg', '.png')):
            filename += '.jpg'
        
        filepath = self.session_dir / subdir / filename
        cv2.imwrite(str(filepath), image)
        self.logger.debug(f"Image saved: {filepath}")
        return filepath
    
    def save_data(self, data: dict, filename: str) -> Path:
        """Save JSON data"""
        filepath = self.session_dir / "data" / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.debug(f"Data saved: {filepath}")
        return filepath
    
    def log_detection_results(self, detections: List[Tuple[int, Dict]]):
        """Log detection results in a structured way"""
        if not detections:
            self.logger.warning("No markers detected in frame")
            return
        
        self.logger.info(f"Detected {len(detections)} markers: {[mid for mid, _ in detections]}")
        
        for marker_id, pose in detections:
            tvec = pose['tvec']
            orientation = pose['orientation']
            distance = pose['distance']
            
            self.logger.debug(
                f"Marker {marker_id:2d}: "
                f"pos=({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}), "
                f"orient=({np.degrees(orientation[0]):.1f}°, {np.degrees(orientation[1]):.1f}°, {np.degrees(orientation[2]):.1f}°), "
                f"dist={distance:.3f}m"
            )

# Global session logger instance
session_logger = None

def get_session_logger() -> SessionLogger:
    """Get or create the global session logger"""
    global session_logger
    if session_logger is None:
        session_logger = SessionLogger()
    return session_logger

# -----------------------------
# Profiling decorator
# -----------------------------
def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_session_logger()
        start_time = time.perf_counter()
        logger.logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # в миллисекундах
            logger.logger.debug(f"Completed {func.__name__}: {execution_time:.2f}ms")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            logger.logger.error(f"Failed {func.__name__} after {execution_time:.2f}ms: {e}")
            raise
    return wrapper
    
CAMERA_MATRIX = np.array([[1369.969777345121, 0.0, 1644.2396729412144], [0.0, 1369.2994306470775, 1285.0255987881847], [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.7328083036253085, -0.28350359410819087, -0.0005540692887780926, -0.0012821343946201554, -0.03738973934016638, 1.0045252415116845, -0.14052222433439687, -0.12285786149816241, 0.008635730474242086, -0.0012304383604311727, 0.00018229224907507545, 0.00016397513864849708])


@profile_time
def capture_image(path: str = "/tmp/frame.jpg") -> np.ndarray:
    logger = get_session_logger()
    logger.logger.info("Starting image capture")
    
    if Path(path).exists():
        try:
            Path(path).unlink()
            logger.logger.debug(f"Deleted existing image at {path}")
        except Exception as e:
            logger.logger.error(f"Failed to delete existing image at {path}: {e}")
    try:
        import subprocess
        
        logger.logger.debug("Attempting rpicam-jpeg capture")
        subprocess.run(
            [
                "rpicam-jpeg",
                "--output",
                path,
                "--timeout",
                "100",
                "--nopreview",
                "--roi",
                "0,0,1,1",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        img = cv2.imread(path)
        logger.logger.info("Successfully captured image using rpicam-jpeg")
        
        # Save original captured image
        logger.save_image(img, "original_capture.jpg")
        
        # Return original image without preprocessing
        return img
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.logger.warning(f"rpicam-jpeg failed: {e}, falling back to OpenCV VideoCapture")
        
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.logger.error("Could not open camera device")
            raise RuntimeError("Could not open camera device")
            
        ok, frame = cap.read()
        cap.release()
        
        if not ok:
            logger.logger.error("Camera capture failed")
            raise RuntimeError("Camera capture failed")
            
        logger.logger.info("Successfully captured image using OpenCV VideoCapture")
        logger.save_image(frame, "opencv_capture.jpg")
        
        return frame


@profile_time
def _rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Convert OpenCV Rodrigues rotation vector to XYZ Euler angles (rad)."""
    logger = get_session_logger()
    logger.logger.debug(f"Converting rotation vector to Euler angles: {rvec.flatten()}")

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        logger.logger.debug("Singular rotation matrix detected")
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    
    euler_angles = (float(x), float(y), float(z))
    logger.logger.debug(f"Euler angles (rad): {euler_angles}")
    return euler_angles


# -----------------------------
# Simple exponential‑moving‑average filter per marker
# -----------------------------
class EMAFilter:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.state: Dict[int, np.ndarray] = {}
        logger = get_session_logger()
        logger.logger.info(f"EMA Filter initialized with alpha={alpha}")

    @profile_time
    def update(self, marker_id: int, tvec: np.ndarray) -> np.ndarray:
        """Return the filtered translation vector."""
        logger = get_session_logger()
        
        if marker_id not in self.state:
            self.state[marker_id] = tvec.copy()
            logger.logger.debug(f"Initialized EMA state for marker {marker_id}: {tvec.flatten()}")
        else:
            old_state = self.state[marker_id].copy()
            self.state[marker_id] = (
                self.alpha * tvec + (1.0 - self.alpha) * self.state[marker_id]
            )
            logger.logger.debug(
                f"Updated EMA for marker {marker_id}: "
                f"raw={tvec.flatten()}, "
                f"old_filtered={old_state.flatten()}, "
                f"new_filtered={self.state[marker_id].flatten()}"
            )
        return self.state[marker_id]



class ArucoTracker:
    def __init__(
        self,
        camera_matrix: np.ndarray = CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DIST_COEFFS,
        marker_size: float = 0.04,
        dictionary: int = aruco.DICT_APRILTAG_36h11,
        ema_alpha: float = 0.25,
    ) -> None:
        logger = get_session_logger()
        logger.logger.info("Initializing ArucoTracker")
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size
        self._dict = aruco.getPredefinedDictionary(dictionary)
        self._params = aruco.DetectorParameters()
        
        # Recommended refinements → less jitter
        self._params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._params.cornerRefinementWinSize = 7
        self._params.cornerRefinementMaxIterations = 100
        self._params.cornerRefinementMinAccuracy = 0.05

        self._filter = EMAFilter(alpha=ema_alpha)

        # 3‑D coordinates of the marker corners in its local frame
        half = marker_size / 2.0
        self._obj_pts = np.array(
            [
                [-half, half, 0],  # top‑left
                [half, half, 0],   # top‑right
                [half, -half, 0],  # bottom‑right
                [-half, -half, 0], # bottom‑left
            ],
            dtype=np.float32,
        )
        
        logger.logger.info(
            f"ArucoTracker initialized: "
            f"marker_size={marker_size}, "
            f"dictionary={dictionary}, "
            f"ema_alpha={ema_alpha}"
        )
        
        # Save calibration data
        calib_data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "marker_size": marker_size,
            "dictionary": dictionary,
            "ema_alpha": ema_alpha,
            "detector_parameters": {
                "cornerRefinementMethod": self._params.cornerRefinementMethod,
                "cornerRefinementWinSize": self._params.cornerRefinementWinSize,
                "cornerRefinementMaxIterations": self._params.cornerRefinementMaxIterations,
                "cornerRefinementMinAccuracy": self._params.cornerRefinementMinAccuracy,
            }
        }
        logger.save_data(calib_data, "calibration_data")

    # Pre‑processing helper (less noise, higher contrast)
    @staticmethod
    @profile_time
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        logger = get_session_logger()
        logger.logger.debug(f"Preprocessing frame: shape={frame.shape}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.logger.debug(f"Converted to grayscale: shape={gray.shape}")
        
        # gray = cv2.fastNlMeansDenoising(gray, h=10)  # gentle denoise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        logger.logger.debug("Applied histogram equalization")
        
        # Save debug image
        logger.save_image(gray, "preprocessed_frame.jpg", "debug_images")
        
        return gray

    @profile_time
    def detect_markers(
        self,
        frame: np.ndarray,
        *,
        draw_axes: bool = False,
        axes_length: float = 0.05,
    ) -> List[Tuple[int, Dict]]:
        logger = get_session_logger()
        logger.logger.info("Starting marker detection")
        
        # Save input frame
        logger.save_image(frame, "input_frame.jpg")
        
        # Preprocess the frame for detection
        gray = self._preprocess(frame)
        logger.save_image(gray, "detection_input.jpg", "debug_images")

        corners, ids, _ = aruco.detectMarkers(gray, self._dict, parameters=self._params)
        
        if ids is None:
            logger.logger.warning("No ArUco markers detected in frame")
            return []

        ids = ids.flatten()
        logger.logger.info(f"Detected {len(ids)} markers: {ids.tolist()}")
        
        results: List[Tuple[int, Dict]] = []
        annotated_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray.copy()

        for idx, marker_id in enumerate(ids):
            logger.logger.debug(f"Processing marker {marker_id}")
            
            img_pts = corners[idx].reshape(4, 2).astype(np.float32)
            logger.logger.debug(f"Marker {marker_id} corners: {img_pts}")

            # Draw marker corners
            cv2.polylines(annotated_frame, [np.int32(img_pts)], True, (0, 255, 0), 2)
            cv2.putText(annotated_frame, str(marker_id), 
                       tuple(np.int32(img_pts[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Pose estimation with a more robust PnP + LM refinement
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts,
                img_pts,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            
            if not ok:
                logger.logger.warning(f"PnP solve failed for marker {marker_id}")
                continue

            logger.logger.debug(f"Initial pose for marker {marker_id}: rvec={rvec.flatten()}, tvec={tvec.flatten()}")

            # Levenberg‑Marquardt refinement -> ~30‑40 % less jitter
            cv2.solvePnPRefineLM(
                self._obj_pts,
                img_pts,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )
            
            logger.logger.debug(f"Refined pose for marker {marker_id}: rvec={rvec.flatten()}, tvec={tvec.flatten()}")

            # Low‑pass filter across frames
            tvec_filtered = self._filter.update(int(marker_id), tvec.reshape(3))

            pose = {
                "rvec": rvec.reshape(3),
                "tvec": tvec_filtered,  # smoothed translation
                "tvec_raw": tvec.reshape(3),  # raw translation for debugging
                "orientation": _rotation_to_euler(rvec),
                "distance": float(np.linalg.norm(tvec_filtered)),
            }

            if draw_axes:
                cv2.drawFrameAxes(annotated_frame, self.camera_matrix, self.dist_coeffs, 
                                rvec, tvec_filtered, axes_length)

            results.append((int(marker_id), pose))
            
            logger.logger.debug(
                f"Marker {marker_id} final pose: "
                f"distance={pose['distance']:.3f}m, "
                f"position=({tvec_filtered[0]:.3f}, {tvec_filtered[1]:.3f}, {tvec_filtered[2]:.3f})"
            )

        # Save annotated frame
        logger.save_image(annotated_frame, "annotated_frame.jpg")
        
        # Save detection results
        detection_data = {
            "timestamp": datetime.now().isoformat(),
            "detected_markers": len(results),
            "marker_ids": [mid for mid, _ in results],
            "detection_results": [
                {
                    "marker_id": int(mid),
                    "pose": {
                        "rvec": pose["rvec"].tolist(),
                        "tvec": pose["tvec"].tolist(),
                        "tvec_raw": pose["tvec_raw"].tolist(),
                        "orientation_rad": pose["orientation"],
                        "orientation_deg": [np.degrees(angle) for angle in pose["orientation"]],
                        "distance": pose["distance"]
                    }
                }
                for mid, pose in results
            ]
        }
        logger.save_data(detection_data, f"detection_{int(time.time())}")
        
        logger.log_detection_results(results)
        return results

    # --------------------------------------------------
    # Utility to get relative pose between two detections
    # --------------------------------------------------
    @staticmethod
    @profile_time
    def relative_pose(marker_a: Dict, marker_b: Dict) -> Dict[str, float]:
        logger = get_session_logger()
        logger.logger.debug("Computing relative pose between two markers")
        
        rvec1, tvec1 = marker_a["rvec"].reshape(3), marker_a["tvec"].reshape(3)
        rvec2, tvec2 = marker_b["rvec"].reshape(3), marker_b["tvec"].reshape(3)
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R_rel = R1.T @ R2
        t_rel = R1.T @ (tvec2 - tvec1)
        yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        
        relative_pose_result = {
            "x": -float(t_rel[1]),
            "y": float(t_rel[0]),
            "z": float(t_rel[2]),
            "yaw_deg": float(np.degrees(yaw_rad)),
        }
        
        logger.logger.debug(f"Relative pose computed: {relative_pose_result}")
        return relative_pose_result


# -----------------------------
# High‑level helper
# -----------------------------

@profile_time
def markers_relative_to_home(
    frame: np.ndarray,
    tracker: ArucoTracker,
    *,
    home_id: int = 0,
) -> List[Tuple[int, Dict]]:
    logger = get_session_logger()
    logger.logger.info(f"Computing markers relative to home marker (id={home_id})")
    
    detections = tracker.detect_markers(frame, draw_axes=False)
    if not detections:
        logger.logger.error("No markers detected in the frame")
        raise ValueError("No markers detected in the frame.")

    try:
        home_pose = next(data for mid, data in detections if mid == home_id)
        logger.logger.info(f"Found home marker {home_id}")
    except StopIteration:
        logger.logger.error(f"Home marker id={home_id} not found in detections: {[mid for mid, _ in detections]}")
        raise ValueError(f"Home marker id={home_id} not found.")

    rel = [
        (mid, ArucoTracker.relative_pose(home_pose, data))
        for mid, data in detections
        if mid != home_id
    ]
    
    # Save relative pose data
    relative_data = {
        "timestamp": datetime.now().isoformat(),
        "home_marker_id": home_id,
        "relative_poses": [
            {
                "marker_id": int(mid),
                "relative_pose": pose
            }
            for mid, pose in rel
        ]
    }
    logger.save_data(relative_data, f"relative_poses_{int(time.time())}")
    
    logger.logger.info(f"Computed relative poses for {len(rel)} markers relative to home marker {home_id}")
    for mid, pose in rel:
        logger.logger.debug(f"Marker {mid} relative to home: x={pose['x']:.3f}, y={pose['y']:.3f}, yaw={pose['yaw_deg']:.1f}°")
    
    return rel


if __name__ == "__main__":
    logger = get_session_logger()
    logger.logger.info("=== ArUco Detection Session Started ===")
    
    try:
        logger.logger.info("Initializing ArUco tracker")
        tracker = ArucoTracker(marker_size=0.04, ema_alpha=0.25)

        logger.logger.info("Capturing frame from camera")
        frame = capture_image()

        logger.logger.info("Computing markers relative to home marker")
        res = markers_relative_to_home(frame, tracker, home_id=0)
        
        logger.logger.info("=== Detection Results ===")
        for mid, pose in res:
            logger.logger.info(
                f"Marker {mid}: x={pose['x']:.3f}m, y={pose['y']:.3f}m, z={pose['z']:.3f}m, yaw={pose['yaw_deg']:.1f}°"
            )
        
        # Save summary results
        summary_data = {
            "session_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_markers_detected": len(res) + 1,  # +1 for home marker
                "home_marker_id": 0,
                "relative_markers": len(res),
                "results": [
                    {
                        "marker_id": int(mid),
                        "x": pose['x'],
                        "y": pose['y'], 
                        "z": pose['z'],
                        "yaw_deg": pose['yaw_deg']
                    }
                    for mid, pose in res
                ]
            }
        }
        logger.save_data(summary_data, "session_summary")
        
        logger.logger.info("=== Session Completed Successfully ===")
        print("\n=== Detection Summary ===")
        print(f"Session directory: {logger.session_dir}")
        print(f"Total markers detected: {len(res) + 1}")
        print(f"Home marker ID: 0")
        print(f"Relative markers: {len(res)}")
        
        from pprint import pprint
        print("\nDetailed results:")
        pprint(res)
        
    except Exception as e:
        logger.logger.error(f"Session failed with error: {e}", exc_info=True)
        print(f"Error: {e}")
        print(f"Check logs in: {logger.session_dir}")
        raise

    
