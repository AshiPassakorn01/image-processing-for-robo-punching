import cv2
import numpy as np
import math
from plc_module import PLCController

def process(image_path, ratio=0.35, inner_r=140, outer_r=225): 
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
    cv2.circle(resized_frame, (cx, cy), outer_r, (180,30,255), 1)
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
    donut_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(donut_mask, (cx, cy), outer_r, 255, -1)
    cv2.circle(donut_mask, (cx, cy), inner_r, 0, -1)

    final_mask = cv2.bitwise_and(blue_mask, blue_mask, mask=donut_mask)

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
        
        dx = centroid_x - cx
        dy = centroid_y - cy

        cv2.arrowedLine(resized_frame, (cx, cy), (centroid_x, centroid_y), (0, 0, 240), 2)
        
        cv2.arrowedLine(resized_frame, (cx, cy), (centroid_x, cy), (50, 230, 240), 2) # X component
        cv2.putText(resized_frame, f"x={dx}", (width - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 230, 240), 2)
        
        cv2.arrowedLine(resized_frame, (centroid_x, cy), (centroid_x, centroid_y), (0, 140, 255), 2) # Y component
        cv2.putText(resized_frame, f"y={dy}", (width - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
            
        cv2.putText(resized_frame, f"Centroid: ({centroid_x}, {centroid_y})", (width - 180, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        zone = get_zone(angle_deg)
        cv2.putText(resized_frame, f"Zone: {zone}", (width - 180, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
            
        cv2.putText(resized_frame, f"Angle: {angle_deg:.2f} deg", (width - 180, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else: 
        print(f"Not found blue object in valid region")

    return resized_frame, angle_deg, dx, dy

def get_zone(angle_deg):
    normalized_angle = angle_deg % 360
    zone = int(((normalized_angle + 30) % 360) // 60) + 1
    return zone

if __name__ == "__main__":
    target_image = r"test case\8.png"
    
    plc = PLCController()
    
    if not plc.plcConnect("192.168.3.250"):
        print("Failed to connect to PLC. Exiting.")
        exit(1)
    
    result_img, found_angle, found_dx, found_dy = process(target_image, ratio=0.35)
    
    zonee = get_zone(found_angle) if found_angle is not None else None
    
    # data write to PLC
    if zonee is not None:
        # zonee [1-6] for robot arm zone selection
        plc.write_holding(address=100, value=zonee)
        
        # zonee - 1 [interpolation table for robot arm]
        plc.write_holding(address=102, value=zonee - 1)
        
        #sequcene: M100 ON -- SET START
        plc.write_M(address=100, status=True)  # M100 ON
        
        print(f"Sent to PLC: D100={zonee}, D102={zonee - 1}, M100=ON")
    else:
        print("No zone command due to missing angle")
    # ------------------------------------

    if result_img is not None:
        if found_angle is not None:
            print(f"result: dx={found_dx}, dy={found_dy}, Angle={found_angle:.2f} deg")
            
        cv2.imshow("Result Image", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    plc.plcDisconnect()