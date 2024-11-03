import cv2
from ultralytics import YOLO
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("DJI_20241016090152_0008_V.MP4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Load YOLO model
model = YOLO("Tahryolov8.pt")  # Replace with the correct path to your model file

# Define minimum confidence threshold
MIN_CONFIDENCE = 0.5

# Class names dictionary (modify based on your model's classes)
class_names = {
    0: "Tahr"
    # Add other classes as needed
}

# Define colors
label_bg_color = (119, 0, 200)  # Royal purple background color
text_color = (255, 255, 255)  # white text color

# Dictionary to keep track of unique object IDs and their trackers
tracked_objects = {}
next_object_id = 0  # Counter for new object IDs

def create_tracker():
    """Create and return a new tracker object."""
    return cv2.TrackerCSRT_create()


def assign_id_to_detection(detections):
    """Assign consistent unique IDs to each detection."""
    global next_object_id

    updated_tracked_objects = {}
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])

        # Ignore low-confidence detections
        if confidence < MIN_CONFIDENCE:
            continue

        # Create a new tracker for this detection
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h) format
        new_tracker = create_tracker()
        new_tracker.init(frame, bbox)

        # Check if the object matches an existing tracked object
        match_found = False
        for obj_id, (tracker, prev_class_id) in tracked_objects.items():
            ok, tracked_bbox = tracker.update(frame)
            if ok:
                tracked_x, tracked_y, tracked_w, tracked_h = map(int, tracked_bbox)
                tracked_center = (tracked_x + tracked_w // 2, tracked_y + tracked_h // 2)
                detection_center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

                # Check if the detection is close enough to the tracked object
                if np.linalg.norm(np.array(tracked_center) - np.array(detection_center)) < 50:
                    updated_tracked_objects[obj_id] = (new_tracker, class_id)
                    match_found = True
                    break

        if not match_found:
            # Assign a new ID if no match is found
            updated_tracked_objects[next_object_id] = (new_tracker, class_id)
            next_object_id += 1

    return updated_tracked_objects

def draw_text_with_background(frame, text, position, font_scale=0.5, font_thickness=1):
    """Draw text with a filled background rectangle."""
    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x, y = position

    # Draw the background rectangle
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), label_bg_color, -1)

    # Put the text on top of the background
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

# Process video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing completed or frame could not be read.")
        break

    # Detect objects in the frame
    results = model(frame)
    detections = results[0].boxes

    # Update object tracking and assign IDs
    tracked_objects = assign_id_to_detection(detections)

    # Draw each tracked object with a purple bounding box and label
    for obj_id, (tracker, class_id) in tracked_objects.items():
        ok, tracked_bbox = tracker.update(frame)
        if ok:
            x, y, w, h = map(int, tracked_bbox)
            class_name = class_names.get(class_id, f"Class {class_id}")
            label = f"{class_name}"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (119, 0, 200), 2)  # Purple color for box outline

            # Draw label with background
            draw_text_with_background(frame, label, (x, y - 10), font_scale=0.5, font_thickness=1)

    # Display the total unique object count on the frame
    object_count_text = f"Thars  Count: {len(tracked_objects)}"
    draw_text_with_background(frame, object_count_text, (10, 40), font_scale=1, font_thickness=2)

    # Write the frame with bounding boxes and count overlay
    video_writer.write(frame)

    # Optional: Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()