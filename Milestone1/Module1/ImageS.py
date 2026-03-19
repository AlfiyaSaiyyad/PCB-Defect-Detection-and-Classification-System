
import cv2
import numpy as np
import os


BASE_IMAGE_DIR = r"C:\Users\afsai\Downloads\PCB_DATASET\PCB_DATASET/images"
TEMPLATE_DIR = r"C:\Users\afsai\Downloads\PCB_DATASET\PCB_DATASET/PCB_USED"
OUTPUT_DIR = "PCB_FINAL"

categories = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
os.makedirs(OUTPUT_DIR, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for cat in categories:
    cat_path = os.path.join(BASE_IMAGE_DIR, cat)
    save_path = os.path.join(OUTPUT_DIR, cat)
    os.makedirs(save_path, exist_ok=True)
    print(f"Processing: {cat}")

    for filename in os.listdir(cat_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            prefix = filename.split('_')[0]
            temp_path = os.path.join(TEMPLATE_DIR, f"{prefix}.JPG")
            test_path = os.path.join(cat_path, filename)

            if not os.path.exists(temp_path): continue

            img_t = cv2.imread(temp_path)
            img_i = cv2.imread(test_path)
            img_i = cv2.resize(img_i, (img_t.shape[1], img_t.shape[0]))

            g_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
            g_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
            
            g_t = clahe.apply(g_t)
            g_i = clahe.apply(g_i)

          
            blur_t = cv2.GaussianBlur(g_t, (3, 3), 0)
            blur_i = cv2.GaussianBlur(g_i, (3, 3), 0)
            
            diff = cv2.absdiff(blur_t, blur_i)

          
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) 

        
            kernel = np.ones((2,2), np.uint8)
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res_img = img_i.copy()
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
      
                if area > 25: 
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    
                    if area > 400:
                        color, label = (0, 255, 0), "Major Defect"
                    else:
                        color, label = (0, 0, 255), "Minor Defect"
                    
                    cv2.rectangle(res_img, (x-2, y-2), (x+w+4, y+h+4), color, 2)
                    cv2.putText(res_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imwrite(os.path.join(save_path, f"res_{filename}"), res_img)

print("Processing Complete! Check the output folders.")