import cv2
import os
import xml.etree.ElementTree as ET

ANNOTATIONS_DIR = r"C:\Users\afsai\Downloads\PCB_DATASET\PCB_DATASET/Annotations"
IMAGES_DIR = r"C:\Users\afsai\Downloads\PCB_DATASET\PCB_DATASET/images"
CROP_OUTPUT_DIR = "PCB_ROI"

os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

print("Starting XML-based Extraction...")

for cat in os.listdir(ANNOTATIONS_DIR):
    cat_anno_path = os.path.join(ANNOTATIONS_DIR, cat)
    cat_img_path = os.path.join(IMAGES_DIR, cat)
    
    if not os.path.isdir(cat_anno_path):
        continue

    save_dir = os.path.join(CROP_OUTPUT_DIR, cat)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Extracting labels for: {cat}")

    for xml_file in os.listdir(cat_anno_path):
        if not xml_file.endswith('.xml'): continue

        
        tree = ET.parse(os.path.join(cat_anno_path, xml_file))
        root = tree.getroot()

      
        img_filename = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(cat_img_path, img_filename)

        if not os.path.exists(img_path):
          
            img_path = img_path.replace('.jpg', '.png')
            if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        if img is None: continue

        for i, obj in enumerate(root.findall('object')):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

          
            h, w, _ = img.shape
            crop = img[max(0, ymin-5):min(h, ymax+5), max(0, xmin-5):min(w, xmax+5)]

          
            crop_name = f"{xml_file.split('.')[0]}_obj{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, crop_name), crop)

print(f"\nSuccessfull '{CROP_OUTPUT_DIR}' OUTPUT Saved in folder")