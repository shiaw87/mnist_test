import cv2
import numpy as np
import os
 
# 設定影像路徑
image_path = "digits.png"
 
# 檢查影像是否存在
if not os.path.exists(image_path):
    print(f"❌ 錯誤: 找不到影像 '{image_path}'，請確認檔案是否存在")
    exit()
 
# 讀取影像（確保影像存在，避免 None 錯誤）
img = cv2.imread(image_path)
if img is None:
    print("❌ 無法讀取影像，請確認影像格式是否正確")
    exit()
 
# 轉換為灰階影像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# 將影像切割成 50 行 × 100 列的數字（每個數字 20x20）
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)  # 轉換為 NumPy 陣列
 
# 分割訓練集和測試集
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # 左半部 2500 張圖
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # 右半部 2500 張圖
 
# 建立標籤（每個數字 0-9 各 250 張）
k = np.arange(10)  # 數字 0-9
train_labels = np.repeat(k, 250)[:, np.newaxis]  # 變成 2500 個標籤
test_labels = train_labels.copy()  # 測試標籤與訓練標籤相同
 
# 初始化 KNN 分類器
knn = cv2.ml.KNearest_create()
 
# 訓練 KNN 模型
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
 
# 使用 KNN 進行測試
ret, result, neighbours, dist = knn.findNearest(test, k=5)
 
# 計算準確度
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(f"✅ KNN 模型測試準確度: {accuracy:.2f}%")
 
# 儲存 KNN 訓練數據
np.savez("knn_data.npz", train=train, train_labels=train_labels)
print("✅ KNN 訓練數據已成功儲存為 'knn_data.npz'")
 
# 測試載入數據
with np.load("knn_data.npz") as data:
    loaded_train = data["train"]
    loaded_train_labels = data["train_labels"]
    print(f"✅ 成功載入儲存的數據: 訓練集大小 {loaded_train.shape}, 標籤大小 {loaded_train_labels.shape}")
 
 