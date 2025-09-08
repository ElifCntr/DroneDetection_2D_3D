import cv2
import numpy as np

# Quick test
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imshow("test", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("cv2.imshow is working!")

'''
        # 6. Final proposals + GT on original
        final_display = frame.copy()
        # Draw proposals in green
        for (x, y, w, h) in merged_boxes:
            cv2.rectangle(final_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw GT in red
        for (x, y, w, h) in gts:
            cv2.rectangle(final_display, (x, y), (x + w, y + h), (0, 0, 255), 2)

        final_display = cv2.resize(final_display, None, fx = scale, fy = scale)
'''
