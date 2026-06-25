Pseudo-Code: 
import cv2 
import numpy as np 
import tensorflow as tf 
import RPi.GPIO as GPIO 
import time 
from collections import Counter 
# ===================================== 
# SERVO SETUP 
# ===================================== 
SERVO_PIN = 17 
INITIAL_ANGLE = 90 
GPIO.setmode(GPIO.BCM) 
GPIO.setup(SERVO_PIN, GPIO.OUT) 
pwm = GPIO.PWM(SERVO_PIN, 50) 
pwm.start(0) 
def set_angle(angle): 
duty = angle / 18 + 2.5 
GPIO.output(SERVO_PIN, True) 
pwm.ChangeDutyCycle(duty) 
time.sleep(0.7) 
pwm.ChangeDutyCycle(0) 
# Initialize servo 
set_angle(INITIAL_ANGLE) 
print("Servo initialized") 
# ===================================== 
# LOAD MODEL 
# ===================================== 
interpreter = tf.lite.Interpreter(model_path="model.tflite") 
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details() 
labels = ['cloth', 'metal', 'other', 'paper', 'plastic', 'wood'] 
 
# ===================================== 
# CAMERA 
# ===================================== 
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
input_size = 640 
 
# ===================================== 
# DETECTION VARIABLES 
# ===================================== 
prediction_buffer = [] 
BUFFER_SIZE = 7 
object_locked = False 
servo_busy = False 
last_detection_time = 0 
RESET_DELAY = 2 
 
# ===================================== 
# SERVO ANGLES 
# ===================================== 
angle_map = { 
    "plastic": 30, 
    "metal": 60, 
    "paper": 120, 
    "cloth": 150, 
    "wood": 180 
} 
 
 
# ===================================== 
# MAIN LOOP 
# ===================================== 
while True: 
    ret, frame = cap.read() 
    if not ret: 
        print("Camera Error") 
        break 
    # ===================================== 
    # IMAGE PREPROCESSING 
    # ===================================== 
    img = cv2.resize(frame, (input_size, input_size)) 
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0) 
 
    # ===================================== 
    # MODEL INFERENCE 
    # ===================================== 
    interpreter.set_tensor(input_details[0]['index'], img) 
    interpreter.invoke() 
    output = interpreter.get_tensor(output_details[0]['index']) 
    predictions = output[0] 
    best_label = None 
    best_conf = 0 
 
    # ===================================== 
    # YOLO DETECTION LOOP 
    # ===================================== 
    for pred in predictions: 
        obj_conf = pred[4] 
        # Ignore weak detections 
        if obj_conf < 0.25: 
            continue 
        class_scores = pred[5:] 
        class_id = np.argmax(class_scores) 
        confidence = float( 
            obj_conf * class_scores[class_id] ) 
        if confidence > best_conf: 
            best_conf = confidence 
            best_label = labels[class_id] 
 
    # ===================================== 
    # STORE STABLE PREDICTIONS 
    # ===================================== 
    if best_label and not object_locked: 
        prediction_buffer.append(best_label) 
    # Keep buffer fixed 
    if len(prediction_buffer) > BUFFER_SIZE: 
        prediction_buffer.pop(0) 
 
    # ===================================== 
    # FINAL STABLE DETECTION 
    # ===================================== 
    if ( 
        len(prediction_buffer) == BUFFER_SIZE 
        and not servo_busy 
        and not object_locked 
    ): 
        label_counts = Counter(prediction_buffer) 
        stable_label = label_counts.most_common(1)[0][0] 
        stable_count = label_counts.most_common(1)[0][1] 
        stability_ratio = stable_count / BUFFER_SIZE 
        print("Prediction Buffer:", prediction_buffer) 
        # Strong stability check 
        if stability_ratio >= 0.70: 
 
            print(" \nFINAL DETECTION:", stable_label) 
            object_locked = True 
            servo_busy = True 
 
            # ===================================== 
            # SERVO ACTION 
            # ===================================== 
            if stable_label in angle_map: 
                target_angle = angle_map[stable_label] 
                print("Rotating Servo To:", target_angle) 
                set_angle(target_angle) 
                time.sleep(1.5) 
                print("Returning Servo") 
                set_angle(INITIAL_ANGLE) 
            last_detection_time = time.time() 
            prediction_buffer.clear() 
            servo_busy = False 
 
    # ===================================== 
    # RESET FOR NEXT WASTE 
    # ===================================== 
    current_time = time.time() 
    if ( 
        object_locked 
        and best_label is None 
        and (current_time - last_detection_time > RESET_DELAY) 
     ): 
        print(" \nReady For Next Waste \n") 
 
        object_locked = False 
        prediction_buffer.clear() 
 
    # ===================================== 
    # DISPLAY 
    # ===================================== 
    display_text = "Waiting for Detection" 
    if best_label: 
        display_text = ( 
            f"Detected: {best_label} " 
            f"Conf: {best_conf:.2f}" 
        ) 
    cv2.putText( 
        frame, 
        display_text, 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 255, 0), 
        2 
    ) 
    cv2.putText( 
        frame, 
        "Press Q or ESC to Exit", 
        (20, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 255), 
        2) 
    # ===================================== 
    # SHOW WINDOW 
    # ===================================== 
    cv2.imshow("AI Waste Segregation", frame) 
    key = cv2.waitKey(1) 
    if key == ord('q') or key == 27: 
        break 
 
# ===================================== 
# CLEANUP 
# ===================================== 
print("Closing Program") 
cap.release() 
cv2.destroyAllWindows() 
pwm.stop() 
GPIO.cleanup() 
print("GPIO Cleaned Successfully") 
 