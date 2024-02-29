import cv2
import apriltag

# Function to neatly draw detection boxes
def draw_detection_boxes(image, detections, tag_families):
    for detection in detections:
        # Obtain tag ID and family
        tag_id = detection.tag_id
        tag_family = tag_families[tag_id]
        # Draw box outline
        cv2.line(image, tuple(detection.corners[0].astype(int)), tuple(detection.corners[1].astype(int)), (255,0,0), 2)
        cv2.line(image, tuple(detection.corners[1].astype(int)), tuple(detection.corners[2].astype(int)), (0,255,0), 2)
        cv2.line(image, tuple(detection.corners[2].astype(int)), tuple(detection.corners[3].astype(int)), (0,0,255), 2)
        cv2.line(image, tuple(detection.corners[3].astype(int)), tuple(detection.corners[0].astype(int)), (255,255,255), 2)
        # Draw center point
        center_point = tuple(detection.center.astype(int))
        cv2.circle(image, center_point, 4, (0, 0, 255), -1)
        # Display tag ID and family above the center 
        cv2.putText(image, str(tag_id), (center_point[0] - 10, center_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(image, tag_family, (center_point[0] - 10, center_point[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# Load image (replace 'image.jpg' with your image path)
image = cv2.imread('img.png')

# Convert to grayscale for more efficient detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Specify the tag family you want to detect (e.g., 'tag36h11',  'tag25h9', etc.)
options = apriltag.DetectorOptions(families='tag16h5')
detector = apriltag.Detector(options)

# Detect AprilTags
detections = detector.detect(gray)
print(detections)

# Draw detection boxes on the image
# draw_detection_boxes(image, detections, detector.tag_families)

# Display the result
cv2.imshow('AprilTag Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
