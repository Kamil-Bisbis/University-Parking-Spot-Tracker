<img width="1664" height="1046" alt="image" src="https://github.com/user-attachments/assets/87811990-2681-4429-8cbc-d8564d08e5f9" /># Parking Monitor System

This project monitors parking spots using a camera feed and sends email notifications when the occupancy status changes. It utilizes OpenCV for video processing, YOLOv3 for object detection, and AWS SES for sending emails with images of the parking lot upon detecting changes.

## Files

- `notifier.py`: Handles sending emails via AWS SES.
- `geometry_utils.py`: Contains functions to calculate Intersection over Union (IoU) and match detected objects to monitored parking spots.
- `parking_monitor.py`: Main script that captures video, detects objects, monitors parking spots, and triggers email notifications.

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/parking-monitor-system.git
   cd parking-monitor-system
