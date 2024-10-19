import cv2
import time
from notifier import send_email
from geometry_utils import match_boxes

# Email parameters
EMAIL_CONFIG = {
    'sender': 'your_sender_email@example.com',
    'recipient': 'your_recipient_email@example.com',
    'subject': 'Parking Spot Occupancy Change Detected',
    'body_text': 'The occupancy status of the parking spots has changed.',
    'attachment_path': 'parking_status.jpg',
    'region': 'us-east-1'
}

# Load YOLOv3 model
def load_yolo_model(cfg_path='yolov3.cfg', weights_path='yolov3.weights'):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

# Perform object detection
def detect_objects(net, frame, conf_threshold=0.5, nms_threshold=0.4):
    (H, W) = frame.shape[:2]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Create a blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Process outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, width, height = detection[0:4] * np.array([W, H, W, H])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    final_boxes = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        final_boxes.append([x, y, x + w, y + h])
    return final_boxes

# Draw rectangles on the frame
def draw_boxes(frame, boxes, color, thickness=2):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)

# Main function
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    cv2.namedWindow('Parking Spot Monitor')
    
    # Load YOLO model
    net = load_yolo_model()
    
    # Variables to store parking spots and occupancy status
    parking_spots = []
    occupancy_status = []
    state_changed = False
    aggregation_time = 10  # seconds
    start_time = time.time()
    
    # Mouse callback function to draw parking spots
    def select_parking_spot(event, x, y, flags, param):
        nonlocal parking_spots, temp_spot
        if event == cv2.EVENT_LBUTTONDOWN:
            temp_spot = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            temp_spot.append((x, y))
            parking_spots.append([temp_spot[0][0], temp_spot[0][1], temp_spot[1][0], temp_spot[1][1]])
            temp_spot = []
    
    temp_spot = []
    cv2.setMouseCallback('Parking Spot Monitor', select_parking_spot)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Draw temporary spot
        if len(temp_spot) == 1:
            window_rect = cv2.getWindowImageRect('Parking Spot Monitor')
            if window_rect:
                window_width, window_height = window_rect[2], window_rect[3]
                cv2.rectangle(frame, temp_spot[0], (window_width, window_height), (255, 0, 0), 2)
    
        # Draw parking spots
        draw_boxes(frame, parking_spots, (0, 255, 0))
    
        # Perform object detection every aggregation_time seconds
        current_time = time.time()
        if (current_time - start_time) >= aggregation_time:
            detected_boxes = detect_objects(net, frame)
            new_occupancy_status = match_boxes(parking_spots, detected_boxes)
            if new_occupancy_status != occupancy_status:
                state_changed = True
                occupancy_status = new_occupancy_status.copy()
            start_time = current_time
    
        # If occupancy state changed, send email notification
        if state_changed:
            # Update frame with occupancy status
            for idx, spot in enumerate(parking_spots):
                color = (0, 0, 255) if occupancy_status[idx] else (0, 255, 0)
                cv2.rectangle(frame, (spot[0], spot[1]), (spot[2], spot[3]), color, 2)
            try:
                # Save the frame as an image
                cv2.imwrite(EMAIL_CONFIG['attachment_path'], frame)
                # Send email
                send_email(
                    EMAIL_CONFIG['sender'],
                    EMAIL_CONFIG['recipient'],
                    EMAIL_CONFIG['subject'],
                    EMAIL_CONFIG['body_text'],
                    EMAIL_CONFIG['attachment_path'],
                    EMAIL_CONFIG['region']
                )
                print("Occupancy state changed. Email sent.")
            except Exception as e:
                print(f"Error sending email: {e}")
            state_changed = False
    
        # Display the frame
        cv2.imshow('Parking Spot Monitor', frame)
    
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
