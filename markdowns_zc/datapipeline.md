# data table
here is the table record the path of data input direclty and how we process them 
| Data Type      | File Path/Source               | Processing Steps                                      |
|----------------|-------------------------------|-------------------------------------------------------|
| Initial input data | /mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/temp/sub_merged_caption.json
/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/all_related_text_form_image.json
/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/filter_qiuping_keword_deduplicated_CN_v1.0.0_decay_lc.json | 1. Load JSON files, this json files contain image and correlated context textual information 2. this is the only data source that we need to prepare |



**notice** all the work file for data pipeline is already recorded and uploaded to gitlab
firstly download the data pipeline code from gitlab
 ```bash
 git clone http://10.70.222.233:11000/mllm/datapipeline.git
 cd datapipeline
 ```



# data pipeline 
## 1. prepare the data and merge the multiple images 
- we need to process this step on ali yun and here is the merge images file path you can run the file on directorty:

./caption_generation/merge_images.py

run with command 

```bash
python merge_images.py --base_file /mnt/storage/MLLM/karol/merge/xxx.json --out_root ./temp_files
```
the base file is the image caption json file path you can change it to your own file path (the Initial input data table first row file path)

- after run the merge_images.py file you will get a directory store like this structure 

```plaintext
./temp_files/
    pdf_name1/
        img1.png
        img2.png
        ...
    pdf_name2/
        img1.png
        img2.png
        ...
    ...
```
the images in such directory is the merged images that we need to process for the next step 
## 2. update the input files, and substitute the image path to the merged image path 
```bash
import os 
import json 
# the path of merged images
merged_images_path = "/mnt/workspace/MLLM/karol/merge_sub_images/merged_subimages/max_selected"
# original json file path
existed_json_files = "filter_qiuping_keword_deduplicated_CN_v1.0.0_decay_lc.json"

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
# save the processed data to a new json file
with open("./sub_merged_caption.json", "w") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
print("Finished processing and saved to sub_merged_caption.json")

```
## 3 move the image to a new directory 
if we process all these images on Ali yun and we need to move the images to local disk for the next step 

```bash
import json 
import os 
import shutil 
import tqdm
# the merged json file path
JSON_PATH1 = "/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/temp/sub_merged_caption_2.json"
JSON_PATH2 = "/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/temp/sub_merged_caption.json"
total_data = json.load(open(JSON_PATH1))
total_data2 = json.load(open(JSON_PATH2))
total_data.extend(total_data2)
selected_img_path = "./img_selected"
os.makedirs(selected_img_path, exist_ok=True)
for sample in tqdm.tqdm(total_data):
    img_path = sample["images"][0]
    img_copy_dir = img_path.split("/")[-2] if "max_selected" in img_path else img_path.split("/")[-3]
    os.makedirs(os.path.join(selected_img_path,img_copy_dir), exist_ok=True)
    if os.path.exists(os.path.join(selected_img_path,img_copy_dir, os.path.basename(img_path))):
        continue    
    shutil.copy(img_path, os.path.join(selected_img_path,img_copy_dir, os.path.basename(img_path)))
print("Finished copying images.")
```
- after run the code you will get a directory like this structure 

```plaintext
./img_selected/
    pdf_name1/
        img1.png
        img2.png
        ...
    pdf_name2/
        img1.png
        img2.png
        ...
    ...
```
the images in such directory is the merged images that we need to process for the next step
## 4. run the data process files, after that, every thing is easy we just need to run files on data pipeline and every thing is done

### generate caption
1. run this command to generation initial caption using Doubao
```bash
python caption_generation/doubao_initial_caption.py --read_path ./sub_merged_caption.json --save_path ./caption_generation/temp/doubao_initial_caption.json 
```
2. run this command to generate refined caption 
```bash
python caption_generation/pipeline_gemini_expand_count_tokens.py --data ./caption_generation/temp/doubao_initial_caption.json --img_path ./img_selected --save_path ./caption_generation/temp/gemini_refined_caption.json
```
3. after the expanded caption generation is finished, you just need to run the following command to generate final vqa data json file 
```bash
python pipeline_gemini_build_vqa.py --input_path ./caption_generation/temp/gemini_refined_caption.json --output_path ./temp_images/vqa.json
```
4. FINALLY, we need to distill the cot for the vqa data json file 
```bash
python distill_cot.py --api_key "YOUR_GOOGLE_API_KEY" --input_file ./temp_images/vqa.json --output_file ./temp_images/cot_distilled_output.json
```
- after run all these files you will get the final vqa data json file with cot distilled 
./temp_images/cot_distilled_output.json
