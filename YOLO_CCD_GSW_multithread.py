"""
Cell Detection and Lighting Control System
=========================================
This program captures images from a Gxipy camera, performs object detection using YOLOv10,
and triggers lighting based on the detected object's position. The system displays live
video with detection results and activates specific lighting configurations for a set duration.

Dependencies:
- gxipy (Gxipy SDK for camera control)
- opencv-python (cv2 for image processing and display)
- numpy (for numerical operations)
- ultralytics (for YOLOv10 model)
- threading and concurrent.futures (for multi-threading)
- queue (for image display queue)

Usage Notes:
1. Ensure a Gxipy-compatible camera is connected
2. Place your YOLOv10 model at "your_model.pt" (update the path in the code)
3. Create an "img" folder with lighting images named in the format "({x}, {y}).bmp"
4. Adjust grid_array coordinates according to your specific lighting configuration
"""

import gxipy as gx
import cv2
import time
import numpy as np
from ultralytics import YOLOv10
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Initialize device manager
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_all_device_list()
if dev_num == 0:
    print("No camera devices found")
    exit(1)
ip = dev_info_list[0].get("ip")
print(f"Using camera IP: {ip}")

# Class names for detection
class_names = ['Dcell', 'Lcell']

# Create thread lock to protect shared variables
lock = threading.Lock()

# Shared variable dictionary to store detection and lighting status
detection_data = {
    'key_pressed': False,  # Flag for exit key press
    'dx': 0,  # Width of detected object
    'dy': 0,  # Height of detected object
    'x1': 0,  # Top-left x coordinate of detection
    'y1': 0,  # Top-left y coordinate of detection
    'has_new_detection': False,  # Flag for new detection data
    'is_lighting': False,  # Flag for active lighting task
    'lighting_image': None,  # Image to display for lighting
    'lighting_duration': 0,  # Duration of lighting (ms)
    'lighting_start_time': 0,  # Start time of lighting (timestamp)
    'image_width': 0,  # Width of captured image
    'image_height': 0  # Height of captured image
}

# Create image queue for main thread display (prevents UI blocking)
display_queue = queue.Queue(maxsize=5)

# Load YOLOv10 model
print("Loading YOLO model...")
# Update the model path to your actual model file
model = YOLOv10("your_model.pt")
print("Model loaded successfully")


def capture_callback(raw_image):
    """
    Camera capture callback function. Processes each captured frame:
    - Converts to numpy array
    - Draws grid lines
    - Performs object detection
    - Updates shared detection data
    - Puts processed image in display queue
    """
    if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
        print("Incomplete frame received")
        return

    # Get image dimensions
    height = raw_image.get_height()
    width = raw_image.get_width()

    # Update image size in shared data
    with lock:
        detection_data['image_width'] = width
        detection_data['image_height'] = height

    # Convert raw image to numpy array
    numpy_image = raw_image.get_numpy_array()
    if numpy_image is None:
        print("Failed to get image data")
        return

    # Create copy for display with annotations
    display_image = numpy_image.copy()

    # Draw grid lines on display image
    vertical_spacing = width // 5
    horizontal_spacing = height // 4
    for n in range(0, width + 1, vertical_spacing):
        cv2.line(display_image, (n, 0), (n, height - 1), color=0, thickness=3)
    for m in range(0, height + 1, horizontal_spacing):
        cv2.line(display_image, (0, m), (width - 1, m), color=0, thickness=3)

    # Prepare image for YOLO detection (resize and convert to 3-channel)
    img = cv2.resize(numpy_image, (640, 640))
    img = np.stack((img, img, img), axis=-1)  # Convert grayscale to 3-channel

    # Perform object detection
    results = model.predict(source=img, imgsz=640, conf=0.4, save=False)
    print(results[0].boxes)

    # Process detection results
    if results and results[0].boxes:
        for data in results[0].boxes.data:
            # Skip low confidence detections
            if data[-2].cpu() < 0.05:
                continue

            # Extract detection box data
            x1, y1, x2, y2, conf, cls = np.array(data.cpu()).tolist()

            # Map coordinates back to original image size
            x1 = int(x1 / 640 * width)
            x2 = int(x2 / 640 * width)
            y1 = int(y1 / 640 * height)
            y2 = int(y2 / 640 * height)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            cls = int(cls)

            # Draw detection results on display image
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
            cv2.putText(display_image, class_names[cls], [x1, y1 - 10],
                        color=(255, 0, 0), fontScale=1,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)

            # Update shared detection data with lock protection
            with lock:
                detection_data['dx'] = dx
                detection_data['dy'] = dy
                detection_data['x1'] = x1
                detection_data['y1'] = y1
                detection_data['has_new_detection'] = True

    # Add processed image to display queue (manage queue overflow)
    try:
        display_queue.put_nowait(display_image)
    except queue.Full:
        try:
            display_queue.get_nowait()  # Remove oldest image
            display_queue.put_nowait(display_image)  # Add new image
        except queue.Empty:
            pass


def lighting_task(x, y, duration):
    """
    Lighting control task. Loads and prepares the lighting image,
    updates lighting status in shared data, and manages lighting duration.

    Args:
        x (int): X coordinate for lighting position
        y (int): Y coordinate for lighting position
        duration (int): Lighting duration in milliseconds
    """
    try:
        print(f"Preparing lighting: x={x}, y={y}, duration={duration}ms")
        # Load lighting image from specified path
        image = cv2.imread(f'img/({x}, {y}).bmp')
        if image is None:
            print(f"Failed to read image: img/({x}, {y}).bmp")
            return

        # Update lighting status in shared data
        with lock:
            detection_data['lighting_image'] = image
            detection_data['lighting_duration'] = duration
            detection_data['lighting_start_time'] = time.time()

        print("Lighting preparation completed")
    except Exception as e:
        print(f"Error in lighting operation: {str(e)}")
    finally:
        # Update lighting status when task completes
        with lock:
            detection_data['is_lighting'] = False


# Initialize camera connection
cam = device_manager.open_device_by_ip(ip)
cam_stream = cam.data_stream[0]

# Configure camera trigger mode
cam.TriggerMode.set(gx.GxSwitchEntry.ON)
cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

# Register capture callback function
cam_stream.register_capture_callback(capture_callback)
print('<Starting acquisition>')
cam.stream_on()

# Grid coordinates for lighting positions (adjust according to your setup)
grid_array = [
    [[1020, 340], [1020, 440], [1020, 540], [1020, 640], [1020, 740]],
    [[1135, 340], [1135, 440], [1135, 540], [1135, 640], [1135, 740]],
    [[1250, 340], [1250, 440], [1250, 540], [1250, 640], [1250, 740]],
    [[1365, 340], [1365, 440], [1365, 540], [1365, 640], [1365, 740]]
]

# Create thread pool for handling lighting tasks
executor = ThreadPoolExecutor(max_workers=4)

# Create main display window
cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video", 1920, 1080)
cv2.moveWindow("Live Video", -1920, 0)

try:
    lighting_window_active = False
    lighting_end_time = 0

    while True:
        # Send software trigger to camera
        cam.TriggerSoftware.send_command()

        # Check for exit condition
        with lock:
            if detection_data['key_pressed']:
                break

        # Display live video frame if available
        try:
            # Non-blocking get of latest frame
            display_image = display_queue.get_nowait()
            cv2.imshow("Live Video", display_image)
        except queue.Empty:
            pass

        # Check and handle lighting display
        with lock:
            lighting_image = detection_data['lighting_image']
            lighting_duration = detection_data['lighting_duration']
            lighting_start = detection_data['lighting_start_time']

        if lighting_image is not None:
            # Calculate elapsed and remaining time for lighting
            elapsed = (time.time() - lighting_start) * 1000  # Convert to milliseconds
            remaining = max(0, lighting_duration - elapsed)

            if remaining > 0:
                # Display lighting image
                if not lighting_window_active:
                    cv2.namedWindow("Lighting", cv2.WINDOW_NORMAL)
                    cv2.moveWindow("Lighting", 0, 0)
                    lighting_window_active = True
                cv2.imshow("Lighting", lighting_image)
                lighting_end_time = time.time() + (remaining / 1000)
            else:
                # Close lighting window when duration ends
                if lighting_window_active:
                    cv2.destroyWindow("Lighting")
                    lighting_window_active = False
                with lock:
                    detection_data['lighting_image'] = None

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            with lock:
                detection_data['key_pressed'] = True
                break

        # Process new detection data if available
        with lock:
            if not detection_data['has_new_detection']:
                time.sleep(0.1)  # Short wait for new detection
                continue

            # Read current detection data
            dx = detection_data['dx']
            dy = detection_data['dy']
            x1 = detection_data['x1']
            y1 = detection_data['y1']
            width = detection_data['image_width']
            height = detection_data['image_height']

            # Reset detection flag
            detection_data['has_new_detection'] = False

        # Skip invalid detections
        if dx == 0 or dy == 0 or width == 0 or height == 0:
            print("Skipping invalid detection")
            continue

        # Calculate grid position based on detection
        vertical_spacing = width // 5
        horizontal_spacing = height // 4
        col = (x1 + dx / 2) // vertical_spacing
        row = (y1 + dy / 2) // horizontal_spacing

        # Clamp row and column to valid range
        row = max(0, min(int(row), len(grid_array) - 1))
        col = max(0, min(int(col), len(grid_array[0]) - 1))

        print(f"Detection position: row={row}, column={col}")

        # Get target coordinates from grid
        coord_array = grid_array[row][col]
        x_target = coord_array[0]
        y_target = coord_array[1]

        time.sleep(0.1)  # Short delay to prevent rapid triggering

        # Check if lighting task is already active
        with lock:
            if detection_data['is_lighting']:
                print("Lighting task already active, skipping")
                continue
            else:
                # Set lighting status
                detection_data['is_lighting'] = True

        # Execute lighting task asynchronously
        executor.submit(lighting_task, x_target, y_target, 20000)

except KeyboardInterrupt:
    print("User interrupted")
finally:
    # Clean up resources
    executor.shutdown(wait=True)
    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()
    print("Program terminated")