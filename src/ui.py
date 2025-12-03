from __future__ import annotations

import queue
import time
import threading
import asyncio
import base64
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

from arm_lib import (
    DeviceConnection,
    BaseClient,
    BaseCyclicClient,
    move_to_home,
    move_to_safe_position,
    set_gripper,
    cartesian_action_movement,
)
from aprli_tag_new import ArucoTracker, capture_image, get_session_logger
from tic_tac_toe import find_best_move

# ────────────────────────────────
# Constants from main.py
# ────────────────────────────────
BOARD = 3
CELL = 0.08
FIELD_SIZE = CELL * BOARD
MARKER_SIZE = 0.04
TILE_SIZE = 0.05
HOME_SHIFT = 0.024
CALIBRATION_CROSS_OFFSET = np.array([-0.05, 0])

# ────────────────────────────────
# Helper function for marker annotation
# ────────────────────────────────

def annotate_markers_with_symbols(image, target_marker_id=None):
    print(f"[DEBUG] annotate_markers_with_symbols called with image shape: {image.shape}, target_marker_id: {target_marker_id}")
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        print(f"[DEBUG] Applied preprocessing: grayscale + histogram equalization")
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.1
        
        print(f"[DEBUG] Created ArUco dictionary (DICT_APRILTAG_36h11) and parameters with corner refinement")
        
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            print(f"[DEBUG] Used new ArUco API")
        except AttributeError:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            print(f"[DEBUG] Used old ArUco API")
        
        print(f"[DEBUG] ArUco detection result: marker_ids={marker_ids}, corners_count={len(marker_corners) if marker_corners else 0}")
        
        if marker_ids is not None and len(marker_ids) > 0:
            print(f"Found {len(marker_ids)} markers: {marker_ids.flatten()}")
            
            for i, marker_id in enumerate(marker_ids.flatten()):
                corners = marker_corners[i][0]
                
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                
                min_x = int(np.min(corners[:, 0]))
                max_x = int(np.max(corners[:, 0]))
                min_y = int(np.min(corners[:, 1]))
                max_y = int(np.max(corners[:, 1]))
                
                marker_width = max_x - min_x
                marker_height = max_y - min_y
                marker_size = int(min(marker_width, marker_height))
                
                thickness = max(8, marker_size // 6)
                
                print(f"Drawing on marker {marker_id} at ({center_x}, {center_y}), size: {marker_width}x{marker_height}")
                
                if marker_id % 2 == 1:
                    color = (0, 0, 255)
                    
                    p0, p1, p2, p3 = corners.astype(int)
                    cv2.line(image, p0, p2, color, thickness)
                    cv2.line(image, p1, p3, color, thickness)

                    print(f"Drew red X on marker {marker_id}")
                elif marker_id > 0 and marker_id % 2 == 0:
                    color = (255, 0, 0)
                    rect = cv2.minAreaRect(corners)
                    (cx, cy), (w, h), angle = rect
                    cv2.ellipse(image, (int(cx), int(cy)),
                                (int(w/2), int(h/2)),
                                angle, 0, 360, color, thickness)

                    print(f"Drew blue O on marker {marker_id}")
                    
                border_thickness = 2
                border_color = (0, 255, 0)
                
                if target_marker_id is not None and marker_id == target_marker_id:
                    border_thickness = 12
                    border_color = (0, 255, 0)
                    print(f"Drawing thick green border for target marker {marker_id}")
                
                cv2.polylines(image, [corners.astype(int)], True, border_color, border_thickness)
        else:
            print("No ArUco markers detected in image")
            
    except Exception as e:
        print(f"Error in annotate_markers_with_symbols: {e}")
        import traceback
        traceback.print_exc()
    
    return image


# ────────────────────────────────
# Helper functions from main.py
# ────────────────────────────────


def pose_to_cell(pose):
    print(pose)
    x = -(pose["x"] + HOME_SHIFT)
    y = -(pose["y"] + HOME_SHIFT)

    if x > FIELD_SIZE or x < 0 or y > FIELD_SIZE or y < 0:
        print(x, y)
        print(f"[INFO] Marker outside the field: x={x:.3f}, y={y:.3f}")
        return None

    col = int(x // CELL)
    row = int(y // CELL)

    if 0 <= col < BOARD and 0 <= row < BOARD:
        return row, col
    return None


def get_marker_type(marker_id):
    if marker_id % 2 == 1:
        return "X"
    elif marker_id > 0 and marker_id % 2 == 0:
        return "O"
    return None


def base_position(base, base_cyclic, dx=0, dy=0, dtz=0):
    cartesian_action_movement(
        base, base_cyclic, x=-0.05 + dx, y=-0.15 + dy, z=-0.4, tx=90, tz=dtz
    )


def find_marker_by_type(
    tracker: ArucoTracker,
    frame,
    *,
    marker_type: str,  # 'X' or 'O'
    home_id: int = 0,
):
    try:
        detections = tracker.detect_markers(frame)
        if not detections:
            raise ValueError("No markers detected in the frame.")

        try:
            home_pose = next(data for mid, data in detections if mid == home_id)
        except StopIteration:
            raise ValueError(f"Home marker id={home_id} not visible.")
        
        for mid, data in detections:
            if get_marker_type(mid) != marker_type:
                continue

            rel_pose = ArucoTracker.relative_pose(home_pose, data)
            x = rel_pose["x"]
            y = rel_pose["y"]
            z = rel_pose["z"]
            yaw = rel_pose["yaw_deg"]

            if x > -FIELD_SIZE and x < 0 and y > -FIELD_SIZE and y < 0:
                continue

            print(
                f"[INFO] marker {mid} ({marker_type}) is found "
                f"x = {x:.3f} "
                f"y = {y:.3f} "
                f"z = {z:.3f} "
                f"z_angle={yaw:.1f}°"
            )
            return mid, x, y, z, yaw

        raise ValueError(f"Marker type {marker_type} is not visible in the frame.")

    except ValueError as err:
        raise RuntimeError(f"Marker type {marker_type} not found! Error: {err}")


def read_board(frame, tracker, home_id=0):
    from aprli_tag_new import markers_relative_to_home
    logger = get_session_logger()
    logger.logger.info(f"Reading board state with home marker id={home_id}")

    logger.save_image(frame, "board_reading_input.jpg")

    rel = markers_relative_to_home(frame, tracker, home_id=home_id)
    board = [["." for _ in range(BOARD)] for _ in range(BOARD)]

    logger.logger.info(f"Found {len(rel)} markers relative to home")
    detected_markers = []

    for mid, pose in rel:
        logger.logger.debug(f"Processing marker {mid}: pose={pose}")
        cell = pose_to_cell(pose)
        if cell is None:
            logger.logger.debug(f"Marker {mid} is outside the board area")
            continue

        r, c = cell
        marker_type = get_marker_type(mid)
        if marker_type:
            board[r][c] = marker_type
            detected_markers.append(
                {
                    "marker_id": mid,
                    "marker_type": marker_type,
                    "position": {"row": r, "col": c},
                    "pose": pose,
                }
            )
            logger.logger.debug(
                f"Placed marker {mid} ({marker_type}) at position ({r}, {c})"
            )
        else:
            logger.logger.debug(f"Marker {mid} has unknown type")

    return board


# ────────────────────────────────
# FastAPI Models
# ────────────────────────────────

class GameState(BaseModel):
    board: List[List[str]]
    human_symbol: str
    robot_symbol: str
    game_over: bool
    winner: Optional[str] = None
    hint: Optional[Tuple[int, int]] = None
    robot_connected: bool
    robot_making_move: bool

class MoveRequest(BaseModel):
    symbol: str

class SymbolChoice(BaseModel):
    symbol: str

# ────────────────────────────────
# WebSocket Connection Manager
# ────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

    async def send_json(self, data: dict):
        message = json.dumps(data)
        await self.send_message(message)

    def add_frame(self, frame_data: str):
        """Thread-safe method to add frame to queue"""
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            pass

    async def send_frames(self):
        """Background task to send frames to WebSocket clients"""
        while True:
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get_nowait()
                    await self.send_json({
                        "type": "frame",
                        "data": frame_data
                    })
                await asyncio.sleep(0.1)  # 10 FPS
            except Exception as e:
                print(f"Error sending frames: {e}")
                await asyncio.sleep(0.1)

manager = ConnectionManager()

# ────────────────────────────────
#  Model (остается без изменений)
# ────────────────────────────────


class TicTacToeModel:
    def __init__(self):
        self.board_changed_callbacks = []
        self.game_over_callbacks = []
        self.reset()

    def reset(self):
        self.board: List[List[str]] = [["" for _ in range(3)] for _ in range(3)]
        self._notify_board_changed()

    def add_board_changed_callback(self, callback):
        self.board_changed_callbacks.append(callback)

    def add_game_over_callback(self, callback):
        self.game_over_callbacks.append(callback)

    def _notify_board_changed(self):
        for callback in self.board_changed_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in board changed callback: {e}")

    def _notify_game_over(self, winner: str):
        for callback in self.game_over_callbacks:
            try:
                callback(winner)
            except Exception as e:
                print(f"Error in game over callback: {e}")

    # helpers ---------------------------------------------------------

    def _lines(self):
        b = self.board
        for i in range(3):
            yield b[i]
            yield [b[0][i], b[1][i], b[2][i]]
        yield [b[0][0], b[1][1], b[2][2]]
        yield [b[0][2], b[1][1], b[2][0]]

    def check_winner(self):
        for line in self._lines():
            if line[0] and all(c == line[0] for c in line):
                return line[0]
        if all(cell for row in self.board for cell in row):
            return "Draw"
        return None

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if not self.board[r][c]]

    def place(self, s: str, r: int, c: int):
        if self.board[r][c]:
            raise ValueError("Cell busy")
        self.board[r][c] = s
        self._notify_board_changed()
        w = self.check_winner()
        if w:
            self._notify_game_over(w)


# ────────────────────────────────
#  Worker threads (адаптированы для FastAPI)
# ────────────────────────────────


class CameraThread(threading.Thread):
    def __init__(self, fps: int = 10):
        super().__init__(daemon=True)
        self._running = True
        self._interval = 1.0 / fps
        self._last_cv_frame = None
        self._target_marker_id = None
        self._frame_callbacks = []

    def add_frame_callback(self, callback):
        self._frame_callbacks.append(callback)

    def run(self):
        while self._running:
            try:
                cv_img = capture_image()
                self._last_cv_frame = cv_img
                
                # Annotate markers with symbols for display (X and O)
                annotated_img = annotate_markers_with_symbols(cv_img.copy(), self._target_marker_id)
                
                # Convert frame to base64 and add to manager queue
                _, buffer = cv2.imencode('.jpg', annotated_img)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                manager.add_frame(frame_b64)
                
                # Notify other callbacks with the annotated frame
                for callback in self._frame_callbacks:
                    try:
                        callback(annotated_img)
                    except Exception as e:
                        print(f"Error in frame callback: {e}")
                    
            except Exception as e:
                print(f"CameraThread error: {e}")
            time.sleep(self._interval)

    def get_last_frame(self):
        import copy
        return copy.deepcopy(self._last_cv_frame) if self._last_cv_frame is not None else None

    def set_target_marker(self, marker_id: Optional[int]):
        self._target_marker_id = marker_id

    def stop(self):
        self._running = False


class RobotThread(threading.Thread):
    def __init__(
        self, model: TicTacToeModel, base, base_cyclic, camera_thread: CameraThread
    ):
        super().__init__(daemon=True)
        self.model = model
        self._q: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
        self._running = True
        self.base = base
        self.base_cyclic = base_cyclic
        self.camera_thread = camera_thread
        self.tracker = ArucoTracker(marker_size=MARKER_SIZE, ema_alpha=1.0)
        
        # Callback lists
        self.predicted_move_callbacks = []
        self.move_done_callbacks = []
        self.calibration_done_callbacks = []
        self.target_marker_changed_callbacks = []
        self.robot_move_started_callbacks = []
        self.robot_move_failed_callbacks = []

    # Callback management
    def add_predicted_move_callback(self, callback):
        self.predicted_move_callbacks.append(callback)

    def add_move_done_callback(self, callback):
        self.move_done_callbacks.append(callback)

    def add_calibration_done_callback(self, callback):
        self.calibration_done_callbacks.append(callback)

    def add_target_marker_changed_callback(self, callback):
        self.target_marker_changed_callbacks.append(callback)

    def add_robot_move_started_callback(self, callback):
        self.robot_move_started_callbacks.append(callback)

    def add_robot_move_failed_callback(self, callback):
        self.robot_move_failed_callbacks.append(callback)

    def _notify_predicted_move(self, r: int, c: int):
        for callback in self.predicted_move_callbacks:
            callback(r, c)

    def _notify_move_done(self, r: int, c: int, symbol: str):
        for callback in self.move_done_callbacks:
            callback(r, c, symbol)

    def _notify_calibration_done(self):
        print("[DEBUG] _notify_calibration_done called")
        for callback in self.calibration_done_callbacks:
            print(f"[DEBUG] Calling calibration done callback: {callback}")
            callback()

    def _notify_target_marker_changed(self, marker_id: int):
        for callback in self.target_marker_changed_callbacks:
            callback(marker_id)

    def _notify_robot_move_started(self):
        for callback in self.robot_move_started_callbacks:
            callback()

    def _notify_robot_move_failed(self, reason: str):
        for callback in self.robot_move_failed_callbacks:
            callback(reason)

    # API -------------------------------------------------------------

    def calibrate(self):
        print("[DEBUG] calibrate() called, adding calibrate command to queue")
        self._q.put(("calibrate", None))

    def calibration_complete(self):
        """Вызывается после подтверждения пользователем размещения доски"""
        self._q.put(("calibration_complete", None))

    def make_move(self, symbol: str):
        self._q.put(("move", symbol))

    def stop(self):
        self._q.put(("stop", None))

    # loop ------------------------------------------------------------

    def run(self):
        print("[DEBUG] RobotThread run() started")
        while self._running:
            cmd, arg = self._q.get()
            print(f"[DEBUG] RobotThread received command: {cmd}")
            if cmd == "stop":
                self._running = False
            elif cmd == "calibrate":
                print("[DEBUG] Processing calibrate command")
                self._do_calibration()
            elif cmd == "calibration_complete":
                self._do_calibration_complete()
            elif cmd == "move":
                self._do_move(arg)  # type: ignore[arg‑type]

    # internals -------------------------------------------------------

    def _do_calibration(self):
        print("[DEBUG] Starting calibration sequence")
        move_to_home(self.base)
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, z=-0.04)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)
        set_gripper(self.base, 80)
        # НЕ перемещаем руку домой автоматически - ждем подтверждения от пользователя
        print("[DEBUG] Calibration sequence completed, calling _notify_calibration_done")
        self._notify_calibration_done()

    def _do_calibration_complete(self):
        """Завершаем калибровку - возвращаем руку домой"""
        move_to_home(self.base)

    def _do_move(self, symbol: str):
        self._notify_robot_move_started()
        
        # 0. Get current frame from camera
        img = self.camera_thread.get_last_frame()
        if img is None:
            print("[ERROR] No frame available from camera")
            self._notify_robot_move_failed("No camera frame available")
            return
    
        # 1. Read board state
        board_from_cam = read_board(img, self.tracker)
        board_from_cam = [[c.replace(".", "") for c in row] for row in board_from_cam]

        # Sync model with camera state
        self.model.board = board_from_cam
        self.model._notify_board_changed()
        time.sleep(0.1)  # Allow UI to update

        # 2. Check if game is already over (win or draw) - enhanced check
        winner = self.model.check_winner()
        if winner:
            print(f"[DEBUG] Game is already over: {winner}. Robot will not make a move.")
            # Reset robot making move flag and notify game over
            self._notify_robot_move_failed("")  # Empty message to reset flag without error
            self.model._notify_game_over(winner)
            return

        # 3. Count available markers before attempting move
        x_count = sum(row.count('X') for row in board_from_cam)
        o_count = sum(row.count('O') for row in board_from_cam)
        
        # Check if robot has any markers left (assuming max 5 X markers and 4 O markers)
        if (symbol == 'X' and x_count >= 5) or (symbol == 'O' and o_count >= 4):
            print(f"[WARNING] No more {symbol} markers available on board")
            self._notify_robot_move_failed(f"No {symbol} markers available for robot move")
            # Check if this results in a draw
            if len(self.model.available_moves()) == 0:
                self.model._notify_game_over("Draw")
            return

        # 4. Find robot's marker to pick up with limited attempts
        marker_found = False
        max_attempts = 10
        attempt = 0
        
        while not marker_found and attempt < max_attempts:
            attempt += 1
            print(f"[DEBUG] Marker search attempt {attempt}/{max_attempts}")
            
            # Get fresh frame for each attempt
            img = self.camera_thread.get_last_frame()
            if img is None:
                print("[ERROR] No frame available from camera")
                time.sleep(0.5)
                continue
                
            try:
                marker_id, x, y, _, yaw = find_marker_by_type(
                    self.tracker, img, marker_type=symbol
                )
                print(f"Marker is found {marker_id} of type {symbol}")
                
                self._notify_target_marker_changed(marker_id)

                detections = self.tracker.detect_markers(img)
                home_yaw = None
                for mid, data in detections:
                    if mid == 0:  # home marker id
                        home_yaw = data["orientation"][2]
                        print(f"Home marker yaw: {home_yaw:.1f}°")
                        break
                
                marker_found = True

            except RuntimeError as e:
                print(f"[ERROR] Attempt {attempt}: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5)  # Wait before next attempt
                else:
                    print(f"[ERROR] Failed to find marker after {max_attempts} attempts")
                    self._notify_target_marker_changed(-1)
                    self._notify_robot_move_failed("Could not find robot marker")
                    return

        # 5. Find best move
        move = find_best_move(board_from_cam, symbol)

        if move is None:
            print("No valid move found for the robot.")
            self._notify_target_marker_changed(-1)
            self._notify_robot_move_failed("No valid moves available")
            winner = self.model.check_winner()
            if winner:
                self.model._notify_game_over(winner)
            return

        # 6. Execute move
        r, c = move
        self._notify_predicted_move(r, c)

        # Calculate the calibration offset in global coordinates
        adj_home_yaw = home_yaw
        adj_home_yaw += np.pi / 2  # Adjust home yaw to match robot's orientation
        adj_home_yaw *= -1
        rotation_matrix = np.array([
            [np.cos(adj_home_yaw), -np.sin(adj_home_yaw)],
            [np.sin(adj_home_yaw),  np.cos(adj_home_yaw)],
        ])

        local_motion = np.array([x, y])
        print(local_motion)
        print(np.rad2deg(home_yaw))

        # Add calibration compensation for pickup
        motion = rotation_matrix @ (local_motion - CALIBRATION_CROSS_OFFSET)
        print(f"[DEBUG] Motion after calibration compensation: {motion}")  
        # print(yaw)
        # exit()

        global_yaw = yaw - np.rad2deg(home_yaw)
        # while global_yaw > 180:
        #     global_yaw -= 360
        # while global_yaw <= -180:
        #     global_yaw += 360

        # Pick and place sequence
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)
        cartesian_action_movement(self.base, self.base_cyclic, x=motion[0], y=motion[1])
        cartesian_action_movement(self.base, self.base_cyclic, tz=global_yaw)

        set_gripper(self.base, 30)
        cartesian_action_movement(self.base, self.base_cyclic, z=-0.05)
        set_gripper(self.base, 70)
        cartesian_action_movement(self.base, self.base_cyclic, z=0.1)

        move_to_home(self.base)
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)

        target_x = -(HOME_SHIFT + CELL * c + CELL / 2)
        target_y = -(HOME_SHIFT + CELL * r + CELL / 2)
        local_motion = [target_x, target_y]
        motion = rotation_matrix @ (local_motion - CALIBRATION_CROSS_OFFSET)
        print(f"[DEBUG] Placement motion after calibration compensation: {motion}")
        cartesian_action_movement(self.base, self.base_cyclic, x=motion[0])
        cartesian_action_movement(self.base, self.base_cyclic, y=motion[1])
        cartesian_action_movement(self.base, self.base_cyclic, tz = np.rad2deg(-home_yaw))

        cartesian_action_movement(self.base, self.base_cyclic, z=-0.045)
        set_gripper(self.base, 50)
        move_to_home(self.base)

        # 7. Update model
        try:
            # Save previous game state before making the move
            was_game_over = self.model.check_winner() is not None
            
            self.model.place(symbol, r, c)
            
            # Check if game just ended due to this move
            is_game_over_now = self.model.check_winner() is not None
            
            # If game just ended, give UI time to update before game over notification
            if not was_game_over and is_game_over_now:
                print(f"[DEBUG] Game ended due to robot move, waiting before final notifications")
                time.sleep(0.5)
            
            self._notify_move_done(r, c, symbol)
            self._notify_target_marker_changed(-1)
        except ValueError as e:
            print(f"Error placing marker in model: {e}")
            self._notify_target_marker_changed(-1)


# ────────────────────────────────
#  FastAPI App Controller
# ────────────────────────────────

app = FastAPI(title="Tic Tac Toe Robot")

class AppController:
    def __init__(self):
        self.model = TicTacToeModel()
        
        # Очередь для сообщений из других потоков
        self._message_queue = queue.Queue()
        
        # Robot Connection
        self.device_connection = None
        self.router = None
        self.robot_connected = False
        try:
            self.device_connection = DeviceConnection.createTcpConnection()
            self.router = self.device_connection.__enter__()  # Initialize the connection
            base = BaseClient(self.router)
            base_cyclic = BaseCyclicClient(self.router)
            self.robot_connected = True
            
            # Move robot to home position on startup
            print("Moving robot to home position on startup...")
            move_to_home(base)
            print("Robot moved to home position successfully")
            
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            base, base_cyclic = None, None

        # threads
        self.cam_thread = CameraThread()
        if self.robot_connected:
            self.robot_thread = RobotThread(self.model, base, base_cyclic, self.cam_thread)

        # Setup callbacks
        self._setup_callbacks()

        # Start threads
        self.cam_thread.start()
        if self.robot_connected:
            self.robot_thread.start()

        # state
        self._hint_pos: Optional[Tuple[int, int]] = None
        self._target_marker_id: Optional[int] = None
        self.human_symbol = "X"
        self.robot_symbol = "O"
        self.current_view = "menu"  # menu, camera, game
        self.game_over_flag = False
        self.winner = None
        self.robot_making_move = False  # Track if robot is currently making a move

    def set_main_loop(self, loop):
        self._main_loop = loop

    async def process_message_queue(self):
        while True:
            try:
                if not self._message_queue.empty():
                    message = self._message_queue.get_nowait()
                    print(f"[DEBUG] Processing message from queue: {message}")
                    
                    if message.get("type") == "board_changed":
                        await self._send_game_state()
                    else:
                        await manager.send_json(message)
                        
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error processing message queue: {e}")
                await asyncio.sleep(0.1)

    def _setup_callbacks(self):
        self.model.add_board_changed_callback(self._on_board_changed_sync)
        self.model.add_game_over_callback(self._on_game_over_sync)
        
        if self.robot_connected:
            # Connect robot callbacks
            self.robot_thread.add_predicted_move_callback(self._on_predicted_move_sync)
            self.robot_thread.add_calibration_done_callback(self._on_calibration_done_sync)
            self.robot_thread.add_move_done_callback(self._on_robot_move_done_sync)
            self.robot_thread.add_target_marker_changed_callback(self._on_target_marker_changed_sync)
            self.robot_thread.add_robot_move_started_callback(self._on_robot_move_started_sync)
            self.robot_thread.add_robot_move_failed_callback(self._on_robot_move_failed_sync)

    def _on_board_changed_sync(self):
        try:
            self._message_queue.put_nowait({"type": "board_changed"})
        except queue.Full:
            pass

    def _on_game_over_sync(self, winner: str):
        self.game_over_flag = True
        self.winner = winner
        self.robot_making_move = False  # Reset robot move flag when game ends
        try:
            self._message_queue.put_nowait({
                "type": "game_over",
                "winner": winner,
                "message": "Draw!" if winner == "Draw" else f"Winner: {winner}"
            })
        except queue.Full:
            pass

    def _on_predicted_move_sync(self, r: int, c: int):
        self._hint_pos = (r, c)

    def _on_calibration_done_sync(self):
        print("[DEBUG] _on_calibration_done_sync called - sending calibration_done message")
        try:
            self._message_queue.put_nowait({
                "type": "calibration_done",
                "message": "Place the board and press OK."
            })
            print("[DEBUG] Message added to queue successfully")
        except queue.Full:
            print("[ERROR] Message queue is full")

    def _on_robot_move_done_sync(self, r: int, c: int, symbol: str):
        self.robot_making_move = False
        # Only send human turn message if game is not over
        if not self.model.check_winner():
            try:
                self._message_queue.put_nowait({
                    "type": "human_turn",
                    "message": "Place your marker and press 'My turn is done'"
                })
            except queue.Full:
                pass

    def _on_target_marker_changed_sync(self, marker_id: int):
        self._target_marker_id = None if marker_id == -1 else marker_id
        self.cam_thread.set_target_marker(self._target_marker_id)

    def _on_robot_move_started_sync(self):
        self.robot_making_move = True
        try:
            self._message_queue.put_nowait({
                "type": "robot_move_started",
                "message": "Robot is making a move..."
            })
        except queue.Full:
            pass

    def _on_robot_move_failed_sync(self, reason: str):
        self.robot_making_move = False
        try:
            self._message_queue.put_nowait({
                "type": "robot_move_failed",
                "message": f"Robot move failed: {reason}"
            })
        except queue.Full:
            pass

    async def _send_game_state(self):
        state = GameState(
            board=self.model.board,
            human_symbol=self.human_symbol,
            robot_symbol=self.robot_symbol,
            game_over=self.game_over_flag,
            winner=self.winner,
            hint=self._hint_pos,
            robot_connected=self.robot_connected,
            robot_making_move=self.robot_making_move
        )
        await manager.send_json({
            "type": "game_state",
            "data": state.dict()
        })

    # ─── API Methods -------------------------------------------------

    def get_game_state(self) -> GameState:
        return GameState(
            board=self.model.board,
            human_symbol=self.human_symbol,
            robot_symbol=self.robot_symbol,
            game_over=self.game_over_flag,
            winner=self.winner,
            hint=self._hint_pos,
            robot_connected=self.robot_connected,
            robot_making_move=self.robot_making_move
        )

    def run_calibration(self):
        print(f"[DEBUG] run_calibration called, robot_connected: {self.robot_connected}")
        if not self.robot_connected: 
            return {"error": "Robot not connected"}
        print("[DEBUG] Calling robot_thread.calibrate()")
        self.robot_thread.calibrate()
        return {"message": "Robot is moving to grab pose…"}

    def complete_calibration(self):
        if not self.robot_connected:
            return {"error": "Robot not connected"}
        self.robot_thread.calibration_complete()
        return {"message": "Calibration completed, robot returned home"}

    def start_game(self, human_symbol: str):
        if not self.robot_connected: 
            return {"error": "Robot not connected"}
        
        self.human_symbol = human_symbol
        self.robot_symbol = "O" if human_symbol == "X" else "X"
        
        self.model.reset()
        self._hint_pos = None
        self.game_over_flag = False
        self.winner = None
        self.robot_making_move = False
        self.current_view = "game"

        # Check current board state before starting
        img = self.cam_thread.get_last_frame()
        if img is not None:
            try:
                board_from_cam = read_board(img, self.robot_thread.tracker)
                board_from_cam = [[c.replace(".", "") for c in row] for row in board_from_cam]
                
                # Update model with current board state
                self.model.board = board_from_cam
                
                # Check if game is already over before starting
                winner = self.model.check_winner()
                if winner:
                    self.game_over_flag = True
                    self.winner = winner
                    self.model._notify_game_over(winner)
                    return {"error": f"Game already finished: {winner}"}
                    
            except Exception as e:
                print(f"[ERROR] Failed to read initial board state: {e}")

        if self.robot_symbol == "X":
            self.robot_making_move = True
            self.robot_thread.make_move(self.robot_symbol)
            return {"message": "Robot is making the first move"}
        else:
            return {"message": "Your turn! Place your marker and press 'My turn is done'"}

    def trigger_robot_move(self):
        if not self.robot_connected:
            return {"error": "Robot not connected"}
        self.robot_making_move = True
        self.robot_thread.make_move(self.robot_symbol)
        return {"message": "Robot is making a move"}

    def end_game(self):
        self.model.reset()
        self._target_marker_id = None
        self.cam_thread.set_target_marker(None)
        self.current_view = "menu"
        self.game_over_flag = False
        self.winner = None
        self.robot_making_move = False
        return {"message": "Game ended"}

    def shutdown(self):
        # Move robot to safe position before shutdown
        if self.robot_connected:
            try:
                print("Moving robot to safe position before shutdown...")
                move_to_safe_position(self.robot_thread.base)
            except Exception as e:
                print(f"Error moving robot to safe position: {e}")
        
        self.cam_thread.stop()
        if self.robot_connected:
            self.robot_thread.stop()
        if self.device_connection:
            self.device_connection.__exit__(None, None, None)

# Global controller instance
controller = AppController()

# ────────────────────────────────
#  FastAPI Routes
# ────────────────────────────────

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(manager.send_frames())
    asyncio.create_task(controller.process_message_queue())

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Tic Tac Toe Robot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1e1e1e;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
        }
        .left-panel {
            flex: 1;
            min-height: 600px;
        }
        .right-panel {
            width: 300px;
            background-color: #2b2b2b;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 20px;
        }
        .video-container {
            width: 100%;
            height: 400px;
            background-color: #000;
            border: 1px solid gray;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .board {
            display: grid;
            grid-template-columns: repeat(3, 80px);
            grid-template-rows: repeat(3, 80px);
            gap: 2px;
            margin: 20px 0;
            justify-content: center;
        }
        .cell {
            width: 80px;
            height: 80px;
            background-color: #1e1e1e;
            border: 2px solid #555;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            font-weight: bold;
        }
        .cell.x { color: #ff4444; }
        .cell.o { color: #4488ff; }
        .cell.hint { color: #888; background-color: #2a2a2a; border-color: #777; }
        button {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 8px 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin: 5px 0;
        }
        button:hover { background-color: #4a4a4a; }
        button:disabled { background-color: #2a2a2a; color: #666; cursor: not-allowed; }
        .menu {
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
            justify-content: center;
            height: 400px;
        }
        .hidden { display: none; }
        #messages {
            background-color: #1e1e1e;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 10px;
            margin: 10px 0;
            min-height: 50px;
            max-height: 150px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <!-- Menu View -->
            <div id="menu-view" class="menu">
                <h1>Tic Tac Toe Robot</h1>
                <button onclick="showSymbolChoice()">Play</button>
                <button onclick="showCamera()">Check Camera</button>
                <button onclick="calibrate()">Configure</button>
            </div>

            <!-- Camera View -->
            <div id="camera-view" class="hidden">
                <h2>Camera View</h2>
                <div class="video-container" id="camera-container">
                    <img id="camera-frame" style="max-width: 100%; max-height: 100%;" />
                </div>
                <button onclick="showMenu()">Back</button>
            </div>

            <!-- Game View -->
            <div id="game-view" class="hidden">
                <h2>Game</h2>
                <div class="video-container" id="game-camera-container">
                    <img id="game-camera-frame" style="max-width: 100%; max-height: 100%;" />
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div id="game-controls" class="hidden">
                <div class="board" id="board"></div>
                <button onclick="humanMoveDone()">My turn is done</button>
                <button onclick="endGame()">Give up</button>
            </div>
            <div id="messages"></div>
        </div>
    </div>

    <!-- Symbol Choice Modal -->
    <div id="symbol-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;" onclick="hideSymbolChoice()">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #2b2b2b; padding: 30px; border-radius: 10px; text-align: center; max-width: 400px; position: relative;" onclick="event.stopPropagation()">
            <button onclick="hideSymbolChoice()" style="position: absolute; top: 10px; right: 10px; background: transparent; border: none; color: white; font-size: 20px; cursor: pointer; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">&times;</button>
            <h3>Choose your symbol</h3>
            <button onclick="startGame('X')" style="margin: 5px;">X</button>
            <button onclick="startGame('O')" style="margin: 5px;">O</button>
            <button onclick="startGame(Math.random() > 0.5 ? 'X' : 'O')" style="margin: 5px;">Random</button>
        </div>
    </div>

    <!-- Symbol Confirmation Modal -->
    <div id="symbol-confirmation-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #2b2b2b; padding: 30px; border-radius: 10px; text-align: center; max-width: 400px;">
            <h3>You are playing as <span id="chosen-symbol">X</span></h3>
            <p>Ready to start the game?</p>
            <button onclick="confirmStartGame()">Start Game</button>
            <button onclick="hideSymbolConfirmation()">Choose Again</button>
        </div>
    </div>

    <!-- Calibration Modal -->
    <div id="calibration-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #2b2b2b; padding: 30px; border-radius: 10px; text-align: center; max-width: 400px;">
            <h3>Calibration</h3>
            <p id="calibration-message">Press OK when calibration is done</p>
            <button onclick="completeCalibration()">OK</button>
        </div>
    </div>

    <!-- Game Over Modal -->
    <div id="game-over-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #2b2b2b; padding: 30px; border-radius: 10px; text-align: center; max-width: 400px;">
            <h3>Game Over!</h3>
            <p id="game-over-message">Game finished</p>
            <button onclick="hideGameOverModal()">OK</button>
        </div>
    </div>

    <script>
        let ws = null;
        let currentView = 'menu';
        let selectedSymbol = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);
            };
        }

        function handleWebSocketMessage(data) {
            console.log('[DEBUG] Received WebSocket message:', data);
            if (data.type === 'frame') {
                const img1 = document.getElementById('camera-frame');
                const img2 = document.getElementById('game-camera-frame');
                if (img1) img1.src = 'data:image/jpeg;base64,' + data.data;
                if (img2) img2.src = 'data:image/jpeg;base64,' + data.data;
            } else if (data.type === 'game_state') {
                updateGameBoard(data.data);
                updateButtonStates(data.data);
            } else if (data.type === 'game_over') {
                console.log('[DEBUG] Received game_over message:', data.message);
                addMessage('Game Over: ' + data.message);
                showGameOverModal(data.message);
            } else if (data.type === 'calibration_done') {
                console.log('[DEBUG] Received calibration_done message, showing modal');
                // Показываем модальное окно калибровки
                showCalibrationModal();
                addMessage(data.message || 'Calibration ready - please confirm board placement');
            } else if (data.type === 'human_turn') {
                addMessage(data.message);
            } else if (data.type === 'robot_move_started') {
                addMessage(data.message);
            } else if (data.type === 'robot_move_failed') {
                // Only show message if it's not empty (empty means game was already over)
                if (data.message && data.message.trim() !== '') {
                    addMessage(data.message);
                    // If robot failed due to missing markers, show special handling
                    if (data.message.includes('Could not find robot marker') || 
                        data.message.includes('No valid moves available') ||
                        data.message.includes('No') && data.message.includes('markers available')) {
                        showGameOverModal("Robot cannot continue: " + data.message);
                    }
                }
            }
        }

        function updateGameBoard(gameState) {
            const board = document.getElementById('board');
            if (!board) return;
            
            board.innerHTML = '';
            
            for (let r = 0; r < 3; r++) {
                for (let c = 0; c < 3; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    
                    const value = gameState.board[r][c];
                    if (value === 'X') {
                        cell.textContent = 'X';
                        cell.classList.add('x');
                    } else if (value === 'O') {
                        cell.textContent = 'O';
                        cell.classList.add('o');
                    } else if (gameState.hint && gameState.hint[0] === r && gameState.hint[1] === c) {
                        cell.textContent = '•';
                        cell.classList.add('hint');
                    }
                    
                    board.appendChild(cell);
                }
            }
        }

        function updateButtonStates(gameState) {
            const humanMoveDoneBtn = document.querySelector('button[onclick="humanMoveDone()"]');
            const giveUpBtn = document.querySelector('button[onclick="endGame()"]');
            
            if (humanMoveDoneBtn) {
                if (gameState.robot_making_move || gameState.game_over) {
                    humanMoveDoneBtn.disabled = true;
                    if (gameState.robot_making_move) {
                        humanMoveDoneBtn.textContent = 'Robot is thinking...';
                    }
                } else {
                    humanMoveDoneBtn.disabled = false;
                    humanMoveDoneBtn.textContent = 'My turn is done';
                }
            }
            
            if (giveUpBtn) {
                giveUpBtn.disabled = gameState.robot_making_move;
            }
        }

        function addMessage(message) {
            const messages = document.getElementById('messages');
            messages.innerHTML += '<div>' + message + '</div>';
            messages.scrollTop = messages.scrollHeight;
        }

        function showMenu() {
            document.getElementById('menu-view').classList.remove('hidden');
            document.getElementById('camera-view').classList.add('hidden');
            document.getElementById('game-view').classList.add('hidden');
            document.getElementById('game-controls').classList.add('hidden');
            currentView = 'menu';
        }

        function showCamera() {
            document.getElementById('menu-view').classList.add('hidden');
            document.getElementById('camera-view').classList.remove('hidden');
            document.getElementById('game-view').classList.add('hidden');
            document.getElementById('game-controls').classList.add('hidden');
            currentView = 'camera';
        }

        function showGame() {
            document.getElementById('menu-view').classList.add('hidden');
            document.getElementById('camera-view').classList.add('hidden');
            document.getElementById('game-view').classList.remove('hidden');
            document.getElementById('game-controls').classList.remove('hidden');
            currentView = 'game';
        }

        function showSymbolChoice() {
            document.getElementById('symbol-modal').style.display = 'block';
        }

        function hideSymbolChoice() {
            document.getElementById('symbol-modal').style.display = 'none';
        }

        function showSymbolConfirmation(symbol) {
            selectedSymbol = symbol;
            document.getElementById('chosen-symbol').textContent = symbol;
            document.getElementById('symbol-confirmation-modal').style.display = 'block';
        }

        function hideSymbolConfirmation() {
            document.getElementById('symbol-confirmation-modal').style.display = 'none';
            showSymbolChoice(); // Возвращаемся к выбору символа
        }

        function showCalibrationModal() {
            console.log('[DEBUG] showCalibrationModal called');
            const modal = document.getElementById('calibration-modal');
            if (modal) {
                modal.style.display = 'block';
                console.log('[DEBUG] Modal display set to block');
            } else {
                console.error('[ERROR] calibration-modal element not found');
            }
        }

        function hideCalibrationModal() {
            document.getElementById('calibration-modal').style.display = 'none';
        }

        function showGameOverModal(message) {
            console.log('[DEBUG] showGameOverModal called with message:', message);
            document.getElementById('game-over-message').textContent = message;
            const modal = document.getElementById('game-over-modal');
            if (modal) {
                modal.style.display = 'block';
                console.log('[DEBUG] Game over modal shown');
            } else {
                console.error('[ERROR] game-over-modal element not found');
            }
        }

        function hideGameOverModal() {
            document.getElementById('game-over-modal').style.display = 'none';
            showMenu();
        }

        async function startGame(symbol) {
            hideSymbolChoice();
            showSymbolConfirmation(symbol);
        }

        async function confirmStartGame() {
            document.getElementById('symbol-confirmation-modal').style.display = 'none';
            const response = await fetch('/start-game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbol: selectedSymbol})
            });
            const result = await response.json();
            addMessage(result.message || result.error);
            if (!result.error) {
                showGame();
            }
        }

        async function calibrate() {
            const response = await fetch('/calibrate', {method: 'POST'});
            const result = await response.json();
            if (result.error) {
                addMessage(result.error);
            } else {
                addMessage(result.message);
            }
        }

        async function completeCalibration() {
            const response = await fetch('/calibrate-complete', {method: 'POST'});
            const result = await response.json();
            addMessage(result.message || result.error);
            hideCalibrationModal();
        }

        async function humanMoveDone() {
            // Immediately disable buttons to prevent multiple clicks
            const humanMoveDoneBtn = document.querySelector('button[onclick="humanMoveDone()"]');
            const giveUpBtn = document.querySelector('button[onclick="endGame()"]');
            
            if (humanMoveDoneBtn) {
                humanMoveDoneBtn.disabled = true;
                humanMoveDoneBtn.textContent = 'Processing...';
            }
            if (giveUpBtn) {
                giveUpBtn.disabled = true;
            }
            
            const response = await fetch('/human-move-done', {method: 'POST'});
            const result = await response.json();
            addMessage(result.message || result.error);
        }

        async function endGame() {
            const response = await fetch('/end-game', {method: 'POST'});
            const result = await response.json();
            addMessage(result.message || result.error);
            showGameOverModal("You gave up!");
        }

        // Initialize
        connectWebSocket();
        addMessage('Connected to robot interface');
    </script>
</body>
</html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/game-state")
async def get_game_state():
    return controller.get_game_state()

@app.post("/start-game")
async def start_game(choice: SymbolChoice):
    result = controller.start_game(choice.symbol)
    return result

@app.post("/calibrate")
async def calibrate():
    result = controller.run_calibration()
    return result

@app.post("/calibrate-complete")
async def calibrate_complete():
    result = controller.complete_calibration()
    return result

@app.post("/human-move-done")
async def human_move_done():
    result = controller.trigger_robot_move()
    return result

@app.post("/end-game")
async def end_game():
    result = controller.end_game()
    return result

@app.get("/camera-stream")
async def camera_stream():
    def generate():
        while True:
            frame = controller.cam_thread.get_last_frame()
            if frame is not None:
                annotated_img = annotate_markers_with_symbols(frame.copy(), controller._target_marker_id)
                _, buffer = cv2.imencode('.jpg', annotated_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ────────────────────────────────
#  Entry‑point
# ────────────────────────────────

def main() -> None:
    import atexit
    atexit.register(controller.shutdown)
    
    print("Starting Tic Tac Toe Robot Web Interface...")
    print("Open http://duckie-rpi-arm.local:8000 in your browser")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
