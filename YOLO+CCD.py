"""
YOLOv10 + CCD Camera Object Detection
=====================================
This script captures live video from a Daheng CCD camera using gxipy library,
performs real-time object detection using YOLOv10 model, and displays the results.

Detected classes: 'Dcell' and 'Lcell'

Dependencies:
- gxipy (Daheng camera SDK)
- opencv-python
- numpy
- ultralytics (for YOLOv10)

Usage:
1. Ensure the Daheng CCD camera is properly connected and recognized
2. Place your YOLOv10 model weights ('best.pt') in the same directory
3. Run the script: python YOLO+CCD.py
4. Press 'q' to exit the program
"""

import gxipy as gx
import cv2
import time
import numpy as np
from ultralytics import YOLOv10

# Global variables
device_manager = gx.DeviceManager()
detected_classes = ['Dcell', 'Lcell']  # Classes to detect
key_pressed = False  # Flag for exit command
model = None  # YOLO model placeholder


def capture_callback(raw_image):
    """
    Callback function for camera image capture.
    Processes each frame: converts to numpy array, runs YOLO detection,
    draws bounding boxes, and displays the result.

    Args:
        raw_image: Raw image object from the camera
    """
    global key_pressed

    # Check if frame is complete
    if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
        print("Received incomplete frame")
        return

    # Print frame information
    print(f"Frame ID: {raw_image.get_frame_id()}   Height: {raw_image.get_height()}   Width: {raw_image.get_width()}")

    # Convert raw image to numpy array (grayscale)
    numpy_image = raw_image.get_numpy_array()
    frame_height, frame_width = numpy_image.shape
    print(f"Frame shape: {numpy_image.shape}")

    # Resize to YOLO input size (640x640) and convert to 3-channel (YOLO expects RGB)
    resized_img = cv2.resize(numpy_image, (640, 640))
    rgb_img = np.stack((resized_img, resized_img, resized_img), axis=-1)  # Grayscale to 3-channel
    print(f"Processed image shape for YOLO: {rgb_img.shape}")

    # Perform object detection
    results = model.predict(source=rgb_img, imgsz=640, conf=0.4, save=False)
    print("Detection results:", results[0].boxes)

    # Process detection results
    for box_data in results[0].boxes.data:
        # Extract box coordinates, confidence and class
        x1, y1, x2, y2, conf, cls = np.array(box_data.cpu()).tolist()

        # Skip low confidence detections (additional filter)
        if conf < 0.05:
            continue

        # Convert coordinates back to original frame size
        x1 = int(x1 / 640 * frame_width)
        x2 = int(x2 / 640 * frame_width)
        y1 = int(y1 / 640 * frame_height)
        y2 = int(y2 / 640 * frame_height)
        class_id = int(cls)

        print(
            f"Bounding box: ({x1}, {y1}) to ({x2}, {y2}) - Class: {detected_classes[class_id]}, Confidence: {conf:.2f}")

        # Draw bounding box and label on original image
        # Note: Daheng camera images are read-only, so we create a copy
        frame_with_annotations = numpy_image.copy()
        cv2.rectangle(frame_with_annotations, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        cv2.putText(frame_with_annotations,
                    detected_classes[class_id],
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(255, 0, 0),
                    thickness=4)

    # Create and position window (moves to second monitor if available)
    cv2.namedWindow('Live Video')
    cv2.moveWindow("Live Video", -1920, 0)  # Adjust based on your display setup
    cv2.imshow("Live Video", frame_with_annotations if 'frame_with_annotations' in locals() else numpy_image)

    # Check for exit command ('q' key)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        key_pressed = True


def main():
    """Main function to initialize camera, load model, and run acquisition loop"""
    global model, key_pressed

    try:
        # Initialize camera
        print("Searching for connected cameras...")
        dev_num, dev_info_list = device_manager.update_all_device_list()

        if dev_num == 0:
            raise Exception("No cameras detected. Please check connection.")

        # Get first camera's IP (supports IP, ID, or SN connection)
        camera_ip = dev_info_list[0].get("ip")
        print(f"Found camera with IP: {camera_ip}")

        # Load YOLOv10 model
        print("Loading YOLOv10 model...")
        model = YOLOv10("best.pt")  # Ensure model file exists in working directory

        # Open camera connection
        camera = device_manager.open_device_by_ip(camera_ip)
        data_stream = camera.data_stream[0]

        # Register capture callback
        data_stream.register_capture_callback(capture_callback)

        # Configure camera trigger mode
        camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        camera.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        print("Camera configured for software triggering")

        # Start image acquisition
        print("<Starting image acquisition>")
        camera.stream_on()

        # Acquisition loop with software triggering
        while not key_pressed:
            camera.TriggerSoftware.send_command()
            time.sleep(0.5)  # Adjust frame rate here (2 FPS)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Cleanup resources
        print("Cleaning up resources...")
        if 'camera' in locals() and camera is not None:
            camera.stream_off()
            camera.close_device()
        cv2.destroyAllWindows()
        print("Program exited successfully")


if __name__ == "__main__":
    main()