"""Camera capture for RTSP IP cameras, USB (V4L2), Jetson CSI, or video files."""
import cv2


def _rtsp_pipeline_gst(url, dw, dh):
    """Hardware-accelerated H.264 RTSP decode on Jetson via GStreamer."""
    return (
        f"rtspsrc location={url} latency=100 ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, width={dw}, height={dh}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def _csi_pipeline(sensor_id, sw, sh, dw, dh, fps, flip):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={sw}, height={sh}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={dw}, height={dh}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def list_cameras(cfg):
    """Return per-camera dicts with camera_defaults merged in."""
    defaults = cfg.get("camera_defaults", {})
    return [{**defaults, **cam} for cam in cfg["cameras"]]


def get_camera(cfg, name):
    for cam in list_cameras(cfg):
        if cam["name"] == name:
            return cam
    names = [c["name"] for c in list_cameras(cfg)]
    raise SystemExit(f"camera '{name}' not found. Available: {names}")


def open_capture(cam):
    """Open a cv2.VideoCapture for one merged camera dict (see list_cameras)."""
    source = str(cam["source"])
    w, h, fps = int(cam["width"]), int(cam["height"]), int(cam["fps"])

    if source.startswith("rtsp://"):
        use_gst = bool(cam.get("rtsp_hw_decode", False))
        if use_gst:
            cap = cv2.VideoCapture(_rtsp_pipeline_gst(source, w, h), cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source.startswith("csi:"):
        sensor_id = int(source.split(":", 1)[1])
        pipeline = _csi_pipeline(
            sensor_id,
            int(cam["csi_sensor_width"]), int(cam["csi_sensor_height"]),
            w, h, fps, int(cam["flip_method"]),
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif source.startswith("usb:"):
        index = int(source.split(":", 1)[1])
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # treat as a file path
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{source}'. "
            "Check the device is connected and the source string in config.yaml."
        )
    return cap
