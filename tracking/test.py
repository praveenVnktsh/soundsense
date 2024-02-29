import numpy as np
import cv2

# Create a black image
img = np.zeros((300, 400, 3), dtype=np.uint8)

end_point = (250, 150)
# mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
angle = 80  # Angle of the arc
radius = 100  # Radius of the arc
center = (150, 150)
cv2.ellipse(img, center, (100, 100), 0, 0, angle, (0, 255, 0), thickness=2)
cv2.arrowedLine(img, end_point, (end_point[0], end_point[1] - 50), (0, 255, 0), thickness=2, tipLength=1)

# Display the image
cv2.imshow("Curved Arrow", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
