import os, cv2, math

input_dir = "C:/Users/admin/OneDrive - The University of Manchester/Documents/Kaggle Data/Combined Dataset/Original Size - Test"
output_dir = "C:/Users/admin/OneDrive - The University of Manchester/Documents/Kaggle Data/Combined Dataset/Resized - Test"

input_dir = "C:\\Users\\admin\Downloads\\PCB images unedited-20260307T122610Z-3-001\\PCB images unedited"
output_dir = "C:\\Users\\admin\\OneDrive - The University of Manchester\Documents\\Kaggle Data\\WEEE96"

target_size = 96

for i, file in enumerate(os.listdir(input_dir)):
    # if (i%10 == 0):
    
        file_path = os.path.join(input_dir, file)
        #file_path = "C:\\Users\\admin\OneDrive - The University of Manchester\\Documents\\Kaggle Data\\Stuff\\Inference\\pcb.png"

        image = cv2.imread(file_path)

        if image is None:
            continue
        h, w = image.shape[:2]

        scale = target_size / max(h, w)

        new_w = int(scale*w)
        new_h = int(scale*h)

        image_resized = cv2.resize(image, (new_w, new_h))
        pad_x = math.ceil((target_size - new_w) // 2)
        pad_y = math.ceil((target_size - new_h) // 2)

        image_padded = cv2.copyMakeBorder(image_resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, (0,0,0))
        correction =  cv2.resize(image_padded, (target_size,target_size))
        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, correction)


                