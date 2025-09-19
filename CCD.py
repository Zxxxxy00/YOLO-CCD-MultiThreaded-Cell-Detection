"""
CCD Camera Image Acquisition and Display Program

This program demonstrates how to connect to a CCD camera using the gxipy library,
configure software triggering, capture images, and display them in real-time using OpenCV.
Press 'q' to exit the program.

Dependencies:
- gxipy: Library for controlling Galaxy USB3 Vision cameras
- opencv-python (cv2): Library for image processing and display
- numpy: Library for numerical operations on arrays
- time: Library for adding delays

Usage:
1. Ensure the CCD camera is connected and recognized by the system
2. Install required dependencies: pip install gxipy opencv-python numpy
3. Run the script: python CCD.py
4. Press 'q' in the display window to stop acquisition
"""

import gxipy as gx
import cv2
import time
import numpy

# Global flag to control the acquisition loop
key_pressed = False


def capture_callback(raw_image):
    """
    Callback function for image capture events.

    Processes the captured raw image, converts it to a numpy array,
    displays it using OpenCV, and checks for exit command ('q' key).

    Args:
        raw_image: Raw image object from the camera
    """
    global key_pressed

    # Check if the image is complete
    if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
        print("Incomplete frame received")
        return

    # Print frame information
    print(f"Frame ID: {raw_image.get_frame_id()}   Height: {raw_image.get_height()}   Width: {raw_image.get_width()}")

    # Convert raw image to numpy array
    numpy_image = raw_image.get_numpy_array()
    print(f"Image array shape: {numpy_image.shape}")

    # Create and position display window (moves to second monitor if available)
    cv2.namedWindow('Live Video')
    cv2.moveWindow("Live Video", -1920, 0)  # Position at left edge of second monitor

    # Display the image
    cv2.imshow("Live Video", numpy_image)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        key_pressed = True


def main():
    """
    Main function to initialize camera, configure settings, and run acquisition loop.
    """
    try:
        # Initialize device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_all_device_list()

        if dev_num == 0:
            print("No devices found. Exiting.")
            return

        # Get IP address of the first detected device
        # Note: Devices can also be identified by ID or serial number (SN)
        ip = dev_info_list[0].get("ip")
        print(f"Connecting to camera at IP: {ip}")

        # Open the camera by IP address
        cam = device_manager.open_device_by_ip(ip)

        # Get the first data stream from the camera
        cam_stream = cam.data_stream[0]

        # Register the capture callback function
        cam_stream.register_capture_callback(capture_callback)

        # Configure trigger settings
        cam.TriggerMode.set(gx.GxSwitchEntry.ON)  # Enable trigger mode
        cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)  # Set trigger source to software

        # Start image acquisition
        print('<Starting image acquisition>')
        cam.stream_on()

        # Acquisition loop - send software triggers until 'q' is pressed
        while not key_pressed:
            cam.TriggerSoftware.send_command()  # Send software trigger
            time.sleep(0.2)  # Add delay between triggers (adjust as needed)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up resources
        print('<Stopping acquisition and cleaning up>')
        if 'cam' in locals():
            cam.stream_off()
            cam.close_device()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()