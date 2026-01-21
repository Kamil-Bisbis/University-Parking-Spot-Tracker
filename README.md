![Parking Monitor System](https://kamilbisbis.dev/p4.jpg)

# Parking Monitor System

This project monitors parking spots using a fixed camera feed and sends email notifications when the occupancy status changes.

It uses OpenCV for video processing, YOLOv3 for vehicle detection, and AWS SES to send emails with an annotated image whenever a parking spot becomes occupied or free.

## Files

- `parking_monitor.py`  
  Captures video frames, detects vehicles, tracks parking spot occupancy, and triggers notifications.

- `geometry_utils.py`  
  Provides geometric helpers such as Intersection over Union (IoU) and matching detections to parking spots.

- `notifier.py`  
  Sends email notifications through AWS SES and attaches the annotated frame.

## Setup

### Clone the repository

```bash
git clone https://github.com/yourusername/parking-monitor-system.git
cd parking-monitor-system
````

### Install dependencies

```bash
pip install opencv-python numpy boto3
```

### Download YOLOv3 files

Place the following files in the project root:

* `yolov3.cfg`
* `yolov3.weights`
* `coco.names`

### Configure AWS SES

Verify sender and recipient email addresses in AWS SES and configure AWS credentials.

Example:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=your_region
```

Update email addresses in `notifier.py`.

### Define parking spots

In `parking_monitor.py`, define parking spot bounding boxes based on the camera view.

```python
parking_spots = [
    (x1, y1, x2, y2),
    (x1, y1, x2, y2)
]
```

### Run

```bash
python parking_monitor.py
```

The script starts monitoring the video feed and sends an email whenever a parking spot state changes.
