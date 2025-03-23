import cv2
from lib.camera import Camera

def gstreamer_pipeline(id, config, capture_width=1640, capture_height=1232):
    """
    Returns a GStreamer pipeline string for Raspberry Pi 5 using libcamerasrc.
    The pipeline captures video at capture_width x capture_height and then
    converts it to BGR format for OpenCV.
    """
    framerate = config['general']['framerate']
    display_width = config['general']['width']
    display_height = config['general']['height']

    return (
        "libcamerasrc camera=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGR ! appsink"
        % (
            id,
            capture_width,
            capture_height,
            framerate,
            display_width,
            display_height,
        )
    )

def open_capture(id, config):
    """
    Opens a video capture device using the GStreamer pipeline.
    'config' should be a dictionary containing keys in the 'general' section.
    """
    stream = gstreamer_pipeline(id, config)
    capture = Camera(stream, id)

    if not capture.isOpened():
        raise Exception('Could not open video device ' + str(id))

    return capture
