import os 
import cv2
import shutil
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into parts and convert each part to a float
            line_data = [float(num) for num in line.split()]
            data.append(line_data)
    return data

def read_txt(list):
    for i in range(len(list)):
        id.append(list[i][0])
        x_0.append(list[i][1])  
        y_0.append(list[i][2])
        x_1.append(list[i][3])
        y_1.append(list[i][4])
        
def clear(a, b, c, d, e):
    a.clear()
    b.clear()
    c.clear()
    d.clear()
    e.clear()
    
def xyxy2rec(x0,y0,x1,y1, img_w=1024, img_h=1024):
    bbox_width = float(x1) * img_w
    bbox_height = float(y1) * img_h
    center_x = float(x0) * img_w
    center_y = float(y0) * img_h
    min_x, min_y = center_x - (bbox_width / 2), center_y - (bbox_height / 2)
    max_x, max_y = center_x + (bbox_width / 2), center_y + (bbox_height / 2)
    rec = [min_x, min_y, max_x, max_y]
    return rec

def compute_IOU(rec1,rec2): 
    """
    計算兩個矩形框的交並比。
    :param rec1: (x0,y0,x1,y1) (x0,y0)代表矩形左上的頂點，（x1,y1）代表矩形右下的頂點。下同。
    :param rec2: (x0,y0,x1,y1) 
    :return: 交並比IOU. 
    """ 
    left_column_max = max(rec1[0],rec2[0]) 
    right_column_min = min(rec1[2],rec2[2]) 
    up_row_max = max(rec1[1],rec2[1]) 
    down_row_min = min(rec1[3],rec2[3]) 
    #兩矩形無相交區域的情況
    if left_column_max>=right_column_min or down_row_min<=up_row_max: 
        return 0 
    #兩矩形有相交區域的情況
    else: 
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1]) 
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max) 
        return S_cross/(S1+S2-S_cross) 
    
def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)

    #獲取圖像尺寸
    h, w, c= image.shape

    #對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)

    #計算短邊需要增加多上像素寬度使其與長邊等長
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else :
        pass

    # RGB顏色
    BLACK = [0, 0, 0]

     #給圖像增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value= BLACK)

     #調整圖像大小並返回
    return cv2.resize(constant, (height, width))

def crop(img_source, dir_name, event_source, cropped_area, dir_save_root):
    
    ## collect the 5 sequential feeding frames
    used_numbers = []
    num_2 = 0
    for name in event_source:
        to_do_list = []
        error_occurred = False
        if int(name) in used_numbers:
            continue
        else:
            for num in range(5):
                dir_img = os.path.join(img_source, "{:04d}".format(int(name) + num) + '.jpg')
                to_do_list.append(dir_img)
                used_numbers.append(int(name) + num)
        # print(len(to_do_list))
        

        ## crop the frames & save it
        dir_save = os.path.join(dir_save_root, dir_name + '-' + str(num_2)) ## D:\videos\third\drinking\cut\20230626_1450\cropped\20230625_1450-0
        
        if not os.path.isdir(dir_save):
            os.makedirs(dir_save)
            
        for index, imgs in enumerate(to_do_list):
            # print(imgs)
            img = cv2.imread(imgs)
            if img is None:
                print(f"Could not read image: {imgs}")
                error_occurred = True
                # Handle the error, maybe continue to the next image or abort execution.
            else:
                img_cp = img[cropped_area[1]:cropped_area[3], cropped_area[0]:cropped_area[2]]
                # print(os.path.join(dir_save, str(index) + '.jpg'))
                img_rs = resize_image(img_cp, 640, 640)
                cv2.imwrite(os.path.join(dir_save, str(index) + '.jpg') , img_rs) ## D:\videos\third\drinking\cut\20230626_1450\cropped\20230625_1450-0\0.jpg
            num_2 = num_2 + 1
            
            ## delete the event that had insufficient frames
            if error_occurred:
                try:
                    shutil.rmtree(dir_save)
                    print(f"Deleted directory due to error: {dir_save}")
                except OSError as e:
                    print(f"Error: {e.strerror}")
                    

root = '/home/nas/Personal/Johnny/data_batch_backup'
pen = 'fourth'; batch = '20230622-0730'; day = '20230703'
root_path = os.path.join(root, pen, batch, day, 'cropped')
for dir_name in os.listdir(root_path):
    print(dir_name)
    # print(os.listdir(root_path))
    dir_images = os.path.join(root_path, dir_name, 'images') 
    dir_labels = os.path.join(root_path, dir_name, 'labels') 
      
    ## Setting the definition of variables
    list = []; id = []; x_0 = []; y_0 = []; x_1 = []; y_1 = []; num = 0
    img_w = img_h = 1024
    # feeding_area = (729, 843, 920, 1023)
    
    # feeding_area = (82, 846, 282, 1023) #fourth_drinking
    feeding_area = (693, 180, 985, 584) #fourth_feeding

    feeding_thr = 1
    feeding_event = []
    
    # Loop over all .txt files in the labels directory
    for label_file in os.listdir(dir_labels):
        if label_file.endswith('.txt'):
            file_path = os.path.join(dir_labels, label_file)
            # Read the contents of the file and store in a list
            labels_data = read_txt_file(file_path)
            read_txt(labels_data)

            for index, value in enumerate(labels_data):
                rec1 = xyxy2rec(x_0[index], y_0[index], x_1[index], y_1[index])
                rec2 = feeding_area
                Iou = compute_IOU(rec1, rec2)
                if Iou > 0 & int(id[index]) == 0 :
                    num += 1
            if num > feeding_thr:
                ## obtain the feeding frames ex. 0001 (only name, no .jpg or .txt)
                feeding_event.append(os.path.splitext(label_file)[0])
            num = 0
            clear(id, x_0, x_1, y_0, y_1)
        # print(feeding_event)

    ## definition of save dir 
    dir_cropped = os.path.join(root, pen, batch, 'dataset', 'feeding', day) ## \\192.168.23.21\nas-data\Research_Group\Personal\Johnny\data_batch_backup\third\20230624-0730\dataset\day
    if not os.path.isdir(dir_cropped):
        os.makedirs(dir_cropped)
    crop(dir_images, dir_name, feeding_event, feeding_area, dir_cropped)

