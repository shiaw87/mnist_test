import cv2
import numpy as np
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 轉換為 0 或 1 之間的二值影像
    binary = binary // 255  # 將像素值縮放至 0 或 1
    return binary

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
   
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        digit = binary[y:y+h, x:x+w]
        digit_images.append(digit)
   
    return digit_images

def extract_and_resize_digits(digit_images, output_folder="output_digits"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
   
    resized_digits = []
    for i, img in enumerate(digit_images):
        # Step 1: Resize to 18x16 while maintaining aspect ratio
        aspect_ratio = float(img.shape[1]) / float(img.shape[0])
        if aspect_ratio > 1:  # Wider than tall
            new_w = 14
            new_h = int(14 / aspect_ratio)
        else:  # Taller than wide
            new_h = 16
            new_w = int(16 * aspect_ratio)
       
        digit_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
       
        # Step 2: Create a black image of size 20x20
        digit_padded = np.zeros((20, 20), dtype=np.uint8)
       
        # Step 3: Place the resized image at the center
        start_x = (20 - new_w) // 2
        start_y = (20 - new_h) // 2
        digit_padded[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized
        
        # Step 4: Ensure only 0 and 1 (binary image)
        digit_padded = np.where(digit_padded > 0, 1, 0)  # Set pixels >0 to 1 and others to 0
        
        # Save the resized and padded image
        cv2.imwrite(os.path.join(output_folder, f"digit_{i}.png"), digit_padded * 255)  # Save as 0 or 255 (binary)
        resized_digits.append(digit_padded)
   
    return resized_digits

# 主執行流程
def process_image(image_path):
    binary = preprocess_image(image_path)
    digit_images = find_contours(binary)
    extracted_digits = extract_and_resize_digits(digit_images)
    print(f"已成功分割並儲存 {len(extracted_digits)} 個數字影像！")

# 測試圖片
if __name__ == "__main__":
    input_image = "handwritten_img.png"  # 這裡換成你的圖片
    process_image(input_image)
