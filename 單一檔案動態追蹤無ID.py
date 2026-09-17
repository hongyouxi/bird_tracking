import cv2
import numpy as np


#===============================================================
# 以下這個函式是用來調光讓畫面更清楚，電腦更清楚也比較容易偵測到鳥，中間的數值和調法都是經驗試出來的，基本上在不同背景和時段都能讓畫面中
def process_image(image, information_visible,target_brightness=70):# target_contrast=150):#改成用算的0330的是150其他用比例
    """
    1. 測量指定區域 (ROI) 的亮度與對比度
    2. 調整該區域的亮度與對比度，使其符合目標值
    3. 偵測亮度過高的區域並調降
    4. 將調整後的區域合併回原影像
    """
    
    y_min, y_max, x_min, x_max = 200, 750, 0, 1820
    roi_image = image[y_min:y_max, x_min:x_max]  # 擷取 ROI

    # 轉換為 YUV 色彩空間，獲取亮度通道 Y
    yuv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]

    # 計算當前亮度（平均值）和對比度（標準差）
    brightness = np.mean(y_channel)
    contrast = np.std(y_channel)
    if information_visible is True:
        print(f"當前亮度 : {brightness:.2f}  當前對比度 : {contrast:.2f}")

    target_contrast=(100/brightness)*contrast
    if information_visible is True:
        print(f"目標對比 : {target_contrast:.2f}")
    # 計算亮度調整值（beta）
    beta = target_brightness - brightness

    # 計算對比度調整係數（alpha）
    alpha = target_contrast / (contrast + 1e-5)  # 防止除零錯誤
    alpha = max(1.0, min(alpha, 3.0))  # 限制對比度變化範圍
    if information_visible is True:
        print(f"對比度調整係數alpha: {alpha}  (下限1.0，上限3.0)")
    # 應用亮度 & 對比度調整
    adjusted_roi = cv2.convertScaleAbs(roi_image, alpha=alpha, beta=beta)
    yuv = cv2.cvtColor(adjusted_roi, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    y_sorted = np.sort(y_channel.flatten())
    # 取前 10% 的像素值
    top_10_percent = y_sorted[-int(len(y_sorted) * 0.1):]
    # 計算平均亮度
    avg_top_10 = np.mean(top_10_percent)
    after_std=np.std(y_channel)
    if information_visible is True:
        print(f"調後前亮度10%平均 : {avg_top_10:.2f}  調後對比  : {after_std:.2f}")
        print(f"調整後平均亮度 : {np.mean(y_channel):.2f}")
        print('------------------------------------------------')


    # 使用 CLAHE 進一步增強影像（可選）
    lab = cv2.cvtColor(adjusted_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_roi = cv2.merge((l, a, b))
    enhanced_roi = cv2.cvtColor(enhanced_roi, cv2.COLOR_LAB2BGR)

    # --- Step 3: 將調整後的 ROI 合併回原影像 ---
    output_image = image.copy()
    output_image[y_min:y_max, x_min:x_max] = enhanced_roi

    return output_image  # 返回影像




cap = cv2.VideoCapture("晴天/1714863150-1714863164.ts")#開啟的檔案名稱

fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=500, detectShadows=False)#設定動態追蹤器



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
            # 應用背景減除
    frame_h, frame_w, c = frame.shape
    information_visible=False
    frame=process_image(frame,information_visible)
    fgmask = fgbg.apply(frame)
    # 進行形態學處理，移除小的噪音，保留較大的動態區域
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # 先去噪
    # 找到輪廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_rectangles = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            # 必須存成 list 格式供 groupRectangles 使用

            raw_rectangles.append([x, y, w, h])


    # 設定你容許的「像素距離門檻」，例如：兩個碎框只要距離在 frame_w/15 像素內就強行合併
    DISTANCE_THRESHOLD = frame_w/15
    
    merged_rects = []
    used = [False] * len(raw_rectangles)

    for i in range(len(raw_rectangles)):
        if used[i]:
            continue
            
        # 以目前的框作為基礎，建立一個新家族
        family = [raw_rectangles[i]]
        used[i] = True
        
        # 尋找其他有沒有靠得很近的碎框
        for j in range(i + 1, len(raw_rectangles)):
            if not used[j]:
                # 計算兩個框「中心點」之間的像素距離
                cx1, cy1 = raw_rectangles[i][0] + raw_rectangles[i][2]/2, raw_rectangles[i][1] + raw_rectangles[i][3]/2
                cx2, cy2 = raw_rectangles[j][0] + raw_rectangles[j][2]/2, raw_rectangles[j][1] + raw_rectangles[j][3]/2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # 如果小於門檻，代表牠們是同一個物體的碎框，拉進同一個家族
                if dist < DISTANCE_THRESHOLD:
                    family.append(raw_rectangles[j])
                    used[j] = True
        
        # 將這個家族裡的所有碎框，用「包水餃法」融合成一個大框
        fam_array = np.array(family)
        x1 = np.min(fam_array[:, 0])
        y1 = np.min(fam_array[:, 1])
        x2 = np.max(fam_array[:, 0] + fam_array[:, 2])
        y2 = np.max(fam_array[:, 1] + fam_array[:, 3])
        
        merged_rects.append([x1, y1, x2 - x1, y2 - y1])
    # -----------------------------------------------------------------
    detections = []
    
    # --- 6. 遍歷「合併後」的矩形框，繪圖並記錄數據 ---
    for rect in merged_rects:
        x, y, w, h = rect

        # 在畫面上繪製合併後的綠色大外框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 計算合併後的中心點座標 (cx, cy)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        

        



        ###最終結果，要量化就用這個，裡面是 [中心點x, 中心點y, 左上x, 左上y, 右下x, 右下y]，如果你只要中心點就用前兩個數值，如果你要框的範圍就用後四個數值
        detections.append([cx, cy, x + w, y + h])




    frame = cv2.resize(frame, (1280, 720))
    # out.write(frame)
    # 顯示結果
    cv2.imshow('Bird Detection', frame)
    # time.sleep(1/fps)

    key = cv2.waitKey(1) & 0xFF  
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows() 


