import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
 
# 檢查訓練數據是否存在
data_file = 'knn_data.npz'
if not os.path.exists(data_file):
    print(f"❌ 錯誤: 找不到 {data_file}，請先執行 knn_ocr_sample.py 來生成訓練數據")
    exit()
 
# 載入 KNN 訓練數據
with np.load(data_file) as data:
    train = data['train'].astype(np.float32)
    train_labels = data['train_labels'].astype(np.float32)
 
# 初始化 KNN 分類器
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
 
# 初始化變數
input_number, image_number, image_result = [None] * 10, [None] * 10, [None] * 10
test_number, result, result_str = [None] * 10, [None] * 10, [None] * 10
 
# 設定測試圖片名稱（確保副檔名正確）
input_number = [f"digit_{i}.png" for i in range(10)]  # 如果是 PNG，請改成 ".png"
 
# 預測
for i in range(10):
    image_path = f"output_digits/{input_number[i]}"
   
    # 讀取測試影像
    image_number[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   
    # 確保影像存在
    if image_number[i] is None:
        print(f"❌ 無法讀取影像: {image_path}")
        continue
 
    # 轉換測試影像為 400 維向量（20x20 展平成 1D）
    test_number[i] = image_number[i].reshape(-1, 400).astype(np.float32)
 
    # 使用 KNN 進行預測
    ret, result[i], neighbours, dist = knn.findNearest(test_number[i], k=5)
 
    # 建立 64x64 白色背景結果影像
    image_result[i] = np.ones((64, 64, 3), np.uint8) * 255
 
    # 轉換預測結果為整數
    result_str[i] = str(int(result[i][0][0]))
 
    # 在影像上顯示預測結果
    text_color = (0, 255, 0) if int(result[i][0][0]) == i else (255, 0, 0)
    cv2.putText(image_result[i], result_str[i], (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)
 
# 設定顯示的標題
input_name = [f"Input {i}" for i in range(10)]
predict_name = [f"Predict {i}" for i in range(10)]
 
# 顯示結果
plt.figure(figsize=(10, 4))
for i in range(10):
    if image_number[i] is None:
        continue  # 略過讀取失敗的影像
   
    plt.subplot(2, 10, i + 1)
    plt.imshow(image_number[i], cmap='gray')
    plt.title(input_name[i])
    plt.xticks([]), plt.yticks([])
 
    plt.subplot(2, 10, i + 11)
    plt.imshow(image_result[i])
    plt.title(predict_name[i])
    plt.xticks([]), plt.yticks([])
 
plt.show()
 
 