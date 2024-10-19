def calculate_geometry_utils(boxA: list, boxB: list) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA (list): [x1, y1, x2, y2] coordinates of the first box.
        boxB (list): [x1, y1, x2, y2] coordinates of the second box.
    
    Returns:
        float: IoU value.
    """
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def match_boxes(monitor_boxes: list, det_boxes: list, iou_thresh: float = 0.4) -> list:
    """
    Match detected boxes with monitored parking spots based on IoU threshold.
    
    Args:
        monitor_boxes (list): List of monitored parking spot boxes.
        det_boxes (list): List of detected object boxes.
        iou_thresh (float): IoU threshold to consider a match.
    
    Returns:
        list: List of booleans indicating occupancy status for each monitored spot.
    """
    matches = []
    for mbox in monitor_boxes:
        has_match = False
        for dbox in det_boxes:
            if calculate_iou(mbox, dbox) > iou_thresh:
                has_match = True
                break
        matches.append(has_match)
    return matches
