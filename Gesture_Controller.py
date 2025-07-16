import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol
import pyttsx3
import json
import time
import numpy as np

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings
class Gest(IntEnum):
    FIST = 0
    INDEX_RIGHT = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36
    ROTATE = 41
    CUSTOM_1 = 42
    CUSTOM_2 = 43

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
        self.config = self.load_config()
        self.prev_position = None
        self.prev_time = None

    def load_config(self):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                print("Loaded config.json successfully")
        except FileNotFoundError:
            print("config.json not found, using default configuration")
            config = {
                "thresholds": {
                    "pinch_threshold": 0.3,
                    "frame_count": 4,
                    "rotate_threshold": 20.0,
                    "swipe_threshold": 0.01
                },
                "custom_gestures": {}
            }
        if 'swipe_threshold' not in config['thresholds']:
            print("Warning: swipe_threshold missing in config, using default 0.01")
            config['thresholds']['swipe_threshold'] = 0.01
        return config

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result
        if hand_result is None:
            print("No hand detected, returning PALM")
            self.prev_position = None
            self.prev_time = None
            return

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    def get_angle(self, point1, point2):
        x1, y1 = self.hand_result.landmark[point1].x, self.hand_result.landmark[point1].y
        x2, y2 = self.hand_result.landmark[point2].x, self.hand_result.landmark[point2].y
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle

    def get_velocity(self):
        if self.hand_result is None or self.prev_position is None:
            return 0.0
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0
        dt = current_time - self.prev_time
        if dt == 0:
            return 0.0
        current_position = self.hand_result.landmark[9].x
        vx = (current_position - self.prev_position) / dt
        self.prev_position = current_position
        self.prev_time = current_time
        return vx

    def set_finger_state(self):
        if self.hand_result is None:
            self.finger = 0
            return
        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0
        finger_states = []
        for idx, point in enumerate(points):
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist/dist2, 1)
            except ZeroDivisionError:
                ratio = round(dist/0.01, 1)
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1
                finger_states.append(f"Finger {idx+1}: Extended (ratio={ratio:.1f})")
            else:
                finger_states.append(f"Finger {idx+1}: Folded (ratio={ratio:.1f})")
        if self.finger == 14:
            self.finger = Gest.LAST4
            finger_states.append(f"Adjusted finger state: 14 -> {Gest.LAST4}")
        print(f"Finger state: {finger_states}, finger={self.finger}")

    def get_gesture(self):
        if self.hand_result is None:
            print("No hand result, returning PALM")
            return Gest.PALM

        self.set_finger_state()
        current_gesture = Gest.PALM
        print(f"Pre-gesture check: finger={self.finger}")

        if self.finger == Gest.FIRST2:
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            try:
                ratio = dist1/dist2
            except ZeroDivisionError:
                ratio = 0.0
            dz = self.get_dz([8,12])
            print(f"V_GEST check: dist1={dist1:.2f}, dist2={dist2:.2f}, ratio={ratio:.2f}, dz={dz:.2f}")
            if ratio > 0.85:  # Increased from 0.7
                current_gesture = Gest.V_GEST
                print(f"Detected V_GEST: ratio={ratio:.2f}, dz={dz:.2f}, finger={self.finger}")
            else:
                if dz < 0.2:  # Adjusted from 0.15
                    current_gesture = Gest.TWO_FINGER_CLOSED
                    print(f"Detected TWO_FINGER_CLOSED: ratio={ratio:.2f}, dz={dz:.2f}, finger={self.finger}")
                else:
                    current_gesture = Gest.MID
                    print(f"Detected MID: ratio={ratio:.2f}, dz={dz:.2f}, finger={self.finger}")

        elif self.finger == Gest.FIRST2:
            angle = self.get_angle(8, 12)
            if abs(angle) > self.config['thresholds']['rotate_threshold']:
                current_gesture = Gest.ROTATE
                print(f"Detected ROTATE: angle={angle:.2f}, finger={self.finger}")

        elif self.finger in [int(gest['finger_state']) for gest in self.config['custom_gestures'].values()]:
            for gest_name, gest_data in self.config['custom_gestures'].items():
                if self.finger == int(gest_data['finger_state']):
                    current_gesture = Gest[gest_name]
                    print(f"Detected {gest_name}: finger={self.finger}")
                    break

        elif self.finger in [Gest.LAST3, Gest.LAST4]:
            dist = self.get_dist([8,4])
            if dist < 0.05:
                if self.hand_label == HLabel.MINOR:
                    current_gesture = Gest.PINCH_MINOR
                    print(f"Detected PINCH_MINOR: dist={dist:.2f}, finger={self.finger}")
                else:
                    current_gesture = Gest.PINCH_MAJOR
                    print(f"Detected PINCH_MAJOR: dist={dist:.2f}, finger={self.finger}")
            else:
                current_gesture = Gest.LAST4
                print(f"Detected gesture: LAST4, finger={self.finger}")

        elif self.finger == Gest.INDEX:
            vx = self.get_velocity()
            print(f"Velocity: vx={vx:.4f}")
            if vx > self.config['thresholds']['swipe_threshold']:
                current_gesture = Gest.INDEX_RIGHT
                print(f"Detected gesture: INDEX_RIGHT, finger={self.finger}, vx={vx:.4f}")
            else:
                current_gesture = Gest.INDEX
                print(f"Detected gesture: INDEX, finger={self.finger}, vx={vx:.4f}")

        else:
            try:
                gesture_name = Gest(self.finger).name
                current_gesture = Gest(self.finger)
                print(f"Detected gesture: {gesture_name}, finger={self.finger}")
            except ValueError:
                gesture_name = f"Unknown ({self.finger})"
                current_gesture = Gest.PALM
                print(f"Unknown gesture detected, defaulting to PALM: finger={self.finger}")

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
            self.prev_position = None
            self.prev_time = None

        self.prev_gesture = current_gesture
        if self.frame_count >= self.config['thresholds']['frame_count']:
            self.ori_gesture = current_gesture
        return self.ori_gesture

class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    config = None
    last_action_time = 0
    last_gesture = None
    cooldown = 0.3
    action_triggered = False
    prev_cursor_pos = None

    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                print("Loaded config.json successfully")
        except FileNotFoundError:
            print("config.json not found, using default configuration")
            config = {
                "thresholds": {
                    "pinch_threshold": 0.3,
                    "frame_count": 4,
                    "rotate_threshold": 20.0,
                    "swipe_threshold": 0.01
                },
                "custom_gestures": {}
            }
        if 'swipe_threshold' not in config['thresholds']:
            print("Warning: swipe_threshold missing in config, using default 0.01")
            config['thresholds']['swipe_threshold'] = 0.01
        return config

    def getpinchylv(self, hand_result):
        dist = round((self.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    def getpinchxlv(self, hand_result):
        dist = round((hand_result.landmark[8].x - self.pinchstartxcoord)*10,1)
        return dist
    
    def changesystembrightness(self):
        currentBrightnessLv = sbcontrol.get_brightness(display=0)/100.0
        currentBrightnessLv += self.pinchlv/50.0
        if currentBrightnessLv > 1.0:
            currentBrightnessLv = 1.0
        elif currentBrightnessLv < 0.0:
            currentBrightnessLv = 0.0
        sbcontrol.fade_brightness(int(100*currentBrightnessLv), start=sbcontrol.get_brightness(display=0))
    
    def changesystemvolume(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        currentVolumeLv = volume.GetMasterVolumeLevelScalar()
        currentVolumeLv += self.pinchlv/50.0
        if currentVolumeLv > 1.0:
            currentVolumeLv = 1.0
        elif currentVolumeLv < 0.0:
            currentVolumeLv = 0.0
        volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
    
    def scrollVertical(self):
        pyautogui.scroll(120 if self.pinchlv>0.0 else -120)
    
    def scrollHorizontal(self):
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if self.pinchlv>0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    def get_position(self, hand_result):
        point = 8  # Index finger tip
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if self.prev_cursor_pos is None:
            self.prev_cursor_pos = [x, y]
        # Smoother cursor movement (50/50)
        x = int(0.5 * self.prev_cursor_pos[0] + 0.5 * x)
        y = int(0.5 * self.prev_cursor_pos[1] + 0.5 * y)
        # Boundary checks
        x = max(0, min(x, sx - 1))
        y = max(0, min(y, sy - 1))
        self.prev_cursor_pos = [x, y]
        print(f"Cursor position: x={x}, y={y}, landmark_x={position[0]:.3f}, landmark_y={position[1]:.3f}")
        return (x, y)

    def pinch_control_init(self, hand_result):
        self.pinchstartxcoord = hand_result.landmark[8].x
        self.pinchstartycoord = hand_result.landmark[8].y
        self.pinchlv = 0
        self.prevpinchlv = 0
        self.framecount = 0

    def pinch_control(self, hand_result, controlHorizontal, controlVertical):
        if self.framecount == 5:
            self.framecount = 0
            self.pinchlv = self.prevpinchlv
            if self.pinchdirectionflag == True:
                controlHorizontal()
            elif self.pinchdirectionflag == False:
                controlVertical()
        lvx = self.getpinchxlv(hand_result)
        lvy = self.getpinchylv(hand_result)
        if abs(lvy) > abs(lvx) and abs(lvy) > self.config['thresholds']['pinch_threshold']:
            self.pinchdirectionflag = False
            if abs(self.prevpinchlv - lvy) < self.config['thresholds']['pinch_threshold']:
                self.framecount += 1
            else:
                self.prevpinchlv = lvy
                self.framecount = 0
        elif abs(lvx) > self.config['thresholds']['pinch_threshold']:
            self.pinchdirectionflag = True
            if abs(self.prevpinchlv - lvx) < self.config['thresholds']['pinch_threshold']:
                self.framecount += 1
            else:
                self.prevpinchlv = lvx
                self.framecount = 0

    def handle_controls(self, gesture, hand_result, other_hand_gesture=None):
        x, y = None, None
        if gesture != Gest.PALM and hand_result is not None:
            x, y = self.get_position(hand_result)

        current_time = time.time()
        if gesture != Gest.V_GEST:
            if self.action_triggered and gesture == self.last_gesture:
                print(f"Same gesture detected, skipping action: {gesture.name}")
                return
            if current_time - self.last_action_time < self.cooldown:
                print(f"Cooldown active, skipping action for gesture: {gesture.name}, time remaining: {self.cooldown - (current_time - self.last_action_time):.2f}s")
                return
            if gesture not in [Gest.MID, Gest.TWO_FINGER_CLOSED, Gest.V_GEST] and gesture != Gest.PALM and self.last_gesture not in [Gest.PALM, None]:
                print(f"Non-neutral gesture change detected, waiting for PALM: {self.last_gesture.name} -> {gesture.name}")
                return

        if gesture != Gest.FIST and self.grabflag:
            self.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and self.pinchmajorflag:
            self.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and self.pinchminorflag:
            self.pinchminorflag = False

        if gesture == Gest.PINCH_MAJOR and other_hand_gesture == Gest.PINCH_MINOR:
            self.pinch_control_init(hand_result)
            pyautogui.hotkey('ctrl', '+')
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Two-Hand Pinch: Zoom In")
            print(f"Action triggered: Two-Hand Pinch")
            return

        if gesture == Gest.V_GEST:
            self.flag = True
            pyautogui.moveTo(x, y, duration=0.05)
            if self.last_gesture != Gest.V_GEST:
                with open("detected_gestures.txt", "w") as file:
                    file.write("V_GEST: Mouse Cursor Movement Gesture")
            self.last_gesture = gesture
            self.last_action_time = current_time
            print(f"Action triggered: V_GEST at x={x}, y={y}")
            return

        elif gesture == Gest.PALM:
            self.flag = False
            self.last_gesture = gesture
            self.action_triggered = False
            self.prev_cursor_pos = None  # Reset cursor position
            with open("detected_gestures.txt", "w") as file:
                file.write("Neutral Gesture")
            print(f"Neutral gesture detected, resetting state")
            return

        elif gesture == Gest.FIST:
            if not self.grabflag:
                self.grabflag = True
                pyautogui.mouseDown(button="left")
                self.last_action_time = current_time
                self.action_triggered = True
                print(f"Action triggered: FIST mouse down")
            pyautogui.moveTo(x, y, duration=0.05)
            self.last_gesture = gesture
            with open("detected_gestures.txt", "w") as file:
                file.write("FIST: This Gesture selects anything and can move")
            return

        elif gesture == Gest.MID:
            self.prev_cursor_pos = None  # Prevent cursor movement
            pyautogui.click(button='left')
            self.flag = False
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("MID: Left Click")
            print(f"Action triggered: MID left click")
            return

        elif gesture == Gest.INDEX:
            self.prev_cursor_pos = None
            pyautogui.hotkey('ctrl', 'left')
            self.flag = False
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Index: Move Cursor Left")
            print(f"Action triggered: INDEX ctrl+left")
            return

        elif gesture == Gest.INDEX_RIGHT:
            self.prev_cursor_pos = None
            pyautogui.hotkey('ctrl', 'right')
            self.flag = False
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Index Right: Move Cursor Right")
            print(f"Action triggered: INDEX_RIGHT ctrl+right")
            return

        elif gesture == Gest.TWO_FINGER_CLOSED:
            self.prev_cursor_pos = None  # Prevent cursor movement
            pyautogui.click(button='right')
            self.flag = False
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("TWO_FINGER_CLOSED: Right Click")
            print(f"Action triggered: TWO_FINGER_CLOSED right click")
            return

        elif gesture == Gest.PINCH_MINOR:
            if not self.pinchminorflag:
                self.pinch_control_init(hand_result)
                self.pinchminorflag = True
            self.pinch_control(hand_result, self.scrollHorizontal, self.scrollVertical)
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Pinch Minor: Scroll Horizontal/Vertical")
            print(f"Action triggered: PINCH_MINOR")
            return

        elif gesture == Gest.PINCH_MAJOR:
            if not self.pinchmajorflag:
                self.pinch_control_init(hand_result)
                self.pinchmajorflag = True
            self.pinch_control(hand_result, self.changesystembrightness, self.changesystemvolume)
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Pinch Major: Change System Brightness/Volume")
            print(f"Action triggered: PINCH_MAJOR")
            return

        elif gesture == Gest.ROTATE:
            self.prev_cursor_pos = None
            pyautogui.hotkey('ctrl', '+')
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write("Rotate: Zoom In")
            print(f"Action triggered: ROTATE")
            return

        elif gesture in [Gest.CUSTOM_1, Gest.CUSTOM_2]:
            action = self.config['custom_gestures'][gesture.name]['action']
            description = self.config['custom_gestures'][gesture.name]['description']
            self.prev_cursor_pos = None
            exec(action)
            self.last_gesture = gesture
            self.last_action_time = current_time
            self.action_triggered = True
            with open("detected_gestures.txt", "w") as file:
                file.write(f"{gesture.name}: {description}")
            print(f"Action triggered: {gesture.name}")
            return

class GestureController:
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None
    hr_minor = None
    dom_hand = True
    controller = None

    def __init__(self):
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(f"CAM_WIDTH: {GestureController.CAM_WIDTH}, CAM_HEIGHT: {GestureController.CAM_HEIGHT}")
        GestureController.controller = Controller()

    def classify_hands(self, results):
        left, right = None, None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except:
            pass
        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except:
            pass
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)
        frame_time = 1.0 / 30
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,  # Lowered from 0.7
            min_tracking_confidence=0.7,
            static_image_mode=False,
            model_complexity=1
        ) as hands:
            print("Mediapipe Hands initialized successfully")
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                start_time = time.time()
                success, image = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                height, width = image.shape[:2]
                size = min(width, height)
                start_x = (width - size) // 2
                start_y = (height - size) // 2
                image = image[start_y:start_y+size, start_x:start_x+size]
                print(f"ROI: size={size}, start_x={start_x}, start_y={start_y}")
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    self.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)
                    gest_name_major = handmajor.get_gesture()
                    gest_name_minor = handminor.get_gesture()
                    self.controller.handle_controls(gest_name_major, handmajor.hand_result, gest_name_minor)
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    self.controller.handle_controls(Gest.PALM, None)
                    print("No hand detected in frame")
                cv2.imshow('Gesture Controller', image)
                elapsed_time = time.time() - start_time
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                if cv2.waitKey(5) & 0xFF == 13:
                    break
        GestureController.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gc1 = GestureController()
    gc1.start()