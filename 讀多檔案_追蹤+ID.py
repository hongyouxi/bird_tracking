import cv2
import numpy as np
from pathlib import Path
from norfair import Detection, Tracker
from datetime import datetime


#===============================================================
# 以下這個函式是用來調光讓畫面更清楚，電腦更清楚也比較容易偵測到鳥，中間的數值和調法都是經驗試出來的，基本上在不同背景和時段都能讓畫面中
def process_image(image, information_visible,target_brightness=70):
    """
    1. 測量指定區域 (ROI) 的亮度與對比度
    2. 調整該區域的亮度與對比度，使其符合目標值
    3. 偵測亮度過高的區域並調降
    4. 將調整後的區域合併回原影像
    """
    
    y_min, y_max, x_min, x_max = 200, 750, 0, 1820#只調這個範圍，因為巢只在屋簷下，調光很耗計算資源
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
    #除非出問題不然主程式中，是設information_visible=False，不會顯示
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
#=======================================================================================




#======================================================================================
#這部分是設定追蹤蹤器，用來分配給每隻鳥一個ID，不然如果只是把每隻鳥框出，無法持續追蹤每隻鳥的運動軌跡
tracker = Tracker(distance_function="euclidean",
                  distance_threshold=100,
                  hit_counter_max=15,    #hit_counter_max=n，代表消失超過n貞，才會分配新ID
                  initialization_delay=4)  # 連續出現 4 幀才算真正的鳥，直接過濾掉閃爍的背景雜訊
#==============================================================================================




FOLDER_PATH = Path("test_background1")#把要讀的影片放在這個資料夾



#要一次連續讀整個資料夾的影片，所以先把這個資料夾裡的檔案路徑放進一個list
ts_full_paths = []  # 建立空列表
for p in FOLDER_PATH.iterdir():  # 一個一個拿出來
    if p.is_file() and p.suffix == ".ts":  # 檢查是不是 .ts 檔案
        path_string = str(p)  # 轉成字串路徑
        ts_full_paths.append(path_string)  # 塞進列表裡
#===================================================================




finish_amount=0#看完的影片數
big_break=False
for file_path in ts_full_paths:
    finish_amount+=1
    print("-------------------------------------")
    print(f"正在處理影片: {Path(file_path).name}")
    print(f'處理進度{finish_amount}/{len(ts_full_paths)}')

    #這些影片的檔名就是他們錄製的時間，「-」前後試開始和結束，大約差15秒
    #用的是unix時間，可以轉回一般人看得。有了這個可以一次跑大量影片紀錄東西也不用怕時間跑掉因為15秒就校正一次
    unix_time=int((Path(file_path).name).split('-')[0])
    normal_time = datetime.fromtimestamp(unix_time)
    print("影片開始錄製時間:", normal_time)
    #=======================================================
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'本影片貞數：{fps}')

    
    #設定動態追蹤器
    fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=500, detectShadows=False)


    ire=0
    while cap.isOpened():
        ire+=1
        ret, frame = cap.read()
        if not ret:
            break
                # 應用背景減除
        frame_h, frame_w, c = frame.shape

        #這兩行是把圖片拿去調光(亮度和對比)=====================
        information_visible=False
        frame=process_image(frame,information_visible)
        #====================================================

        #動態追蹤：把在動的東西框出來====================================
        fgmask = fgbg.apply(frame)
        # 進行形態學處理，移除小的噪音，保留較大的動態區域
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # 先去噪
        # 找到輪廓
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #====================================================


        #合併框：因為一隻鳥很多部位都在動，會同時產生很多框，要量化時會出錯像是數進巢次數時會多出來
        raw_rectangles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                (x, y, w, h) = cv2.boundingRect(contour)
                # 必須存成 list 格式供 groupRectangles 使用

                raw_rectangles.append([x, y, w, h])
        # 設定你容許的「像素距離門檻」，例如：兩個碎框只要距離在 frame_w/15 像素內就強行合併=======================================
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
        #=====================================================================================================================


        #使用卡爾曼濾波器和匈牙利演算法(基本上就是用速度和位置猜測這個框在下一貞是哪個框)賦予每個框一個固定的ID
        norfair_detections = []
   
        for rect in merged_rects:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            # 在畫面上繪製合併後的綠色大外框
            # 計算合併後的中心點座標 (cx, cy)
            cx = int(x + w / 2)
            cy = int(y + h / 2)
        
            norfair_detections.append(Detection(points=np.array([[cx, cy]])))

        tracked_objects = tracker.update(detections=norfair_detections)


        detection=[]
        for obj in tracked_objects:




            
            
            # 1. 判斷：這一幀是真正的動態偵測，還是盲猜
            # (因為我們設定要消失幾貞後才會刪除ID所以剛消失時電腦會用之前的速度猜它的位置，但我不要這樣，不然鳥進巢消失後ID還在往前飄)
            if obj.age > obj.hit_counter:
                # 如果是盲猜的，我們手動將「要使用的座標」鎖死在最後一次看到的點
                final_points = obj.last_detection.points
            else:
                # 如果是真正的偵測，就正常使用卡爾曼濾波器修正後的平滑中心座標
                final_points = obj.estimate
                







            ###最終結果================================================
            # 拿到這隻鳥被分配到的固定 ID
            bird_id = obj.id 
            #要做量化就用這行的(cx,cy)就是每隻鳥在畫面中的座標
            cx, cy = int(final_points[0][0]), int(final_points[0][1])
            detection.append((bird_id,cx,cy))
            #這行是把每隻鳥的ID和座標放進一個list裡，之後要量化的時候就可以用這個list裡的資料去判斷每隻鳥有沒有進巢等等，不然只是畫在畫面上是看不出來的
            #=============================================================





            
            # 畫紅色中心方塊與 ID
            cv2.rectangle(frame, (cx-4, cy-4), (cx+4, cy+4), (0, 0, 255), -1)
            cv2.putText(frame, f"Bird ID: {bird_id}", (cx, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)
        #==========================================================================================================================



        frame = cv2.resize(frame, (1280, 720))#如果不調大小會用影片原尺寸播放，可能會超出螢幕大小


        # 開啟視窗顯示結果
        cv2.imwrite(f"123/{ire}.jpg", frame)
        cv2.imshow('Bird Detection', frame)

        key = cv2.waitKey(int(1000/fps)) & 0xFF  # 這是要給人看時才需要設這麼大的延遲，不然就設1(1毫秒)用來讀鍵盤輸入，像下一行
        #key = cv2.waitKey(1) & 0xFF
        if key == 27:#esc鍵
            print("使用者按ESC鍵跳過目前影片")
            break
        elif key == ord('q'):
            print("使用者按下 q，跳出所有影片結束播放")
            big_break=True
            break
    print("-------------------------------------")
    if big_break==True:
        break


cap.release()#要釋放資源不然會用掉太多記憶體
cv2.destroyAllWindows() #要關閉視窗，不然整個程式可能會報錯



