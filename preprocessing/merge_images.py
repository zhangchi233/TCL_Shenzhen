import os 
import json 
merged_images_path = "/home/maxzhang/max_selected"
existed_json_files = "/home/maxzhang/datapipeline/merged_subimages/filter_qiuping_keword_deduplicated_CN_v1.0.0_decay_lc.json"

def storage_processed_data(images_path,existed_json_files):
    storage_dict = {}
    for dirs, _, files in os.walk(images_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                img_path = os.path.join(dirs, file)
                img_key = file.split("_")[-4:]
                img_key = "_".join(img_key).split(".")[0]
                img_key2 = dirs.split("/")[-1]
                #print(img_key2, img_key)
                storage_dict[f"{img_key2}_{img_key}"] = img_path
    json_data = json.load(open(existed_json_files, "r"))
    selected_data = []
    for sample in json_data:
        if len(sample["images"]) > 1:
            x1s, y1s, x2s, y2s = [], [], [], []
            for img in sample["images"]:
                x1,y1,x2,y2 = img.split("_")[-4:]
                y2 = y2.split(".")[0]
                x1s.append(int(x1))
                y1s.append(int(y1))
                x2s.append(int(x2))
                y2s.append(int(y2))
            x1_final = min(x1s)
            y1_final = min(y1s)
            x2_final = max(x2s)
            y2_final = max(y2s)  
            img_key = [str(x1_final), str(y1_final), str(x2_final), str(y2_final)]
            
            img_key = "_".join(img_key)
            img_key2 = img.split("/")[-3]
           
            img_key = f"{img_key2}_{img_key}"
            #print(img_key)
            if img_key in storage_dict:
                #print(img_key)
                sample["images"] = storage_dict[img_key]
            else:
                # print(f"Image key {img_key} not found in storage_dict.")
                # print(list(storage_dict.keys())[:10])  # Print first 10 keys for debugging
                continue
                
        else:
            sample["images"] = sample["images"][0]
        if isinstance(sample["images"], str):
            selected_data.append(sample)
        #print(storage_dict.keys())
    
    return selected_data
print("Start processing merged images...")

processed_data = storage_processed_data(merged_images_path, existed_json_files)
print(len(processed_data))
with open("/home/maxzhang/datapipeline/sub_merged_caption.json", "w") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
print("Finished processing and saved to sub_merged_caption.json")

