import cv2
import numpy as np
import math

def process(image_path, ratio=0.35, inner_r=140, outer_r=225): # เพิ่มพารามิเตอร์ inner_r, outer_r
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"cant open: {image_path}")
        return None, None, None, None

    width = int(frame.shape[1] * ratio)
    height = int(frame.shape[0] * ratio)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    # axis drawing   
    cx = width // 2
    cy = height // 2
    radius = int(800 * ratio)

    # draw info box
    cv2.rectangle(resized_frame, (width - 190, 0), (width-3, 110), (100, 100, 100), -1)
    cv2.rectangle(resized_frame, (width - 190, 0), (width-2, 110), (0, 0, 0), 2)
    
    # draw circles and axes
    # draw outer circle (max boundary)
    cv2.circle(resized_frame, (cx, cy), outer_r, (180,30,255), 1)
    # draw inner circle (min boundary)
    cv2.circle(resized_frame, (cx, cy), inner_r, (180,30,255), 1)

    cv2.line(resized_frame, (cx, 0), (cx, height), (255, 50, 50), 2) #  Y
    cv2.line(resized_frame, (0, cy), (width, cy), (255, 50, 50), 2)  #  X
    cv2.circle(resized_frame, (cx, cy), 8, (0, 0, 255), -1)          #  C

    # find blue object
    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    # mask for blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # mask for donut shape (ring)
    # create a blank mask with the same dimensions as the resized frame
    donut_mask = np.zeros((height, width), dtype=np.uint8)
    
    # paint the outer circle (Outer) with white color
    cv2.circle(donut_mask, (cx, cy), outer_r, 255, -1)
    # paint the inner circle (Inner) with black color to create a donut shape
    cv2.circle(donut_mask, (cx, cy), inner_r, 0, -1)

    # merge blue_mask with donut_mask using bitwise AND to keep only the blue areas that are within the donut shape
    # resulting final_mask will have white pixels only where both blue_mask and donut_mask are white (255)
    final_mask = cv2.bitwise_and(blue_mask, blue_mask, mask=donut_mask)

    # find contours in the final_mask to locate the blue object within the valid donut-shaped region
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6 piece sector lines [pizza slice]
    for i in range(6):
        angle = i * 60 - 30
        end_x = int(cx + radius * math.cos(math.radians(angle)))
        end_y = int(cy - radius * math.sin(math.radians(angle))) 
        cv2.line(resized_frame, (cx, cy), (end_x, end_y), (0, 255, 0), 3)
    
    # zone labels    
    for i in range(6):
        angle = i * 60 
        zone_x_label = int(cx + (radius-35) * math.cos(math.radians(angle)))
        zone_y_label = int(cy - (radius-35) * math.sin(math.radians(angle)))
        cv2.circle(resized_frame, (zone_x_label+11, zone_y_label-5), 25, (255, 0, 150), 2) 
        cv2.putText(resized_frame, f"Z{i+1}", (zone_x_label, zone_y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 150), 2)

    # set defult values   
    angle_deg = None
    dx = None
    dy = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (20, 50, 255), 2)
        
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        cv2.circle(resized_frame, (centroid_x, centroid_y), 8, (0, 255, 0), -1)
        
        # calculate vector components
        dx = centroid_x - cx
        dy = centroid_y - cy

        # draw vector and components
        cv2.arrowedLine(resized_frame, (cx, cy), (centroid_x, centroid_y), (0, 0, 240), 2)
        
        cv2.arrowedLine(resized_frame, (cx, cy), (centroid_x, cy), (50, 230, 240), 2) # X component
        cv2.putText(resized_frame, f"x={dx}", (width - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 230, 240), 2)
        
        cv2.arrowedLine(resized_frame, (centroid_x, cy), (centroid_x, centroid_y), (0, 140, 255), 2) # Y component
        cv2.putText(resized_frame, f"y={dy}", (width - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        # find angle ccw from positive X-axis
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
            
        cv2.putText(resized_frame, f"Centroid: ({centroid_x}, {centroid_y})", (width - 180, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # ดึงฟังก์ชัน get_zone (ผมสมมติว่าคุณประกาศไว้ด้านล่างเหมือนโค้ดเก่า)
        zone = get_zone(angle_deg)
        cv2.putText(resized_frame, f"Zone: {zone}", (width - 180, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            
        cv2.putText(resized_frame, f"Angle: {angle_deg:.2f} deg", (width - 180, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else: 
        print(f"Not found blue object in valid region")

    return resized_frame, angle_deg, dx, dy

def get_zone(angle_deg):
    """
    - zone 1: 330 - 30 
    - zone 2: 30 - 90 
    - zone 3: 90 - 150 
    - zone 4: 150 - 210 
    - zone 5: 210 - 270 
    - zone 6: 270 - 330 
    """
    normalized_angle = angle_deg % 360
    
    zone = int(((normalized_angle + 30) % 360) // 60) + 1
    
    return zone

if __name__ == "__main__":
    target_image = r"test case\7.png"
    
    result_img, found_angle, found_dx, found_dy = process(target_image, ratio=0.35)
        
    """zone = get_zone(found_angle) if found_angle is not None else print("cannot determine zone without angle")
    if zone is not None:
        print(f"Zone: {zone}")"""
        
    if result_img is not None:
        if found_angle is not None:
            print(f"ผลลัพธ์: dx={found_dx}, dy={found_dy}, ทำมุม={found_angle:.2f} องศา")
            
        cv2.imshow("Result Image", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()