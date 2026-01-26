from Caption_generation import *
class NewCaptionGenerator(CaptionGenerator):

    async def generate_caption(self,result):
        captions,img,related_text,root,subtitles = result

        
        image_placeHolder = ""
       
        image_placeHolder += "<image>\n "+ "\n".join(subtitles)+"\n"
        initial_prompt = self.config.initial_prompt
        refined_prompt = self.config.refined_prompt
      
        
        caption_title,keys = safe_title_from_caption_line(captions[0])
        #related_text ="\n".join(related_text)
        # print(caption_title,keys)
        #  caption_title = captions[0].split(" ")[2:]
        # caption_title = " ".join(caption_title)
        try:
            img = [img]
            output = await self.vllm_response(initial_prompt.format(title = caption_title,image = image_placeHolder,related_text=related_text),img = img)
            # print("generate caption is")
            # print(caption)

            extracted_info=output["extracted_info"]
            
            output = await self.vllm_response(refined_prompt.format(title = caption_title,image = image_placeHolder,extracted_info=extracted_info),img = img)
            try:
                caption=output["caption"]
            except:
                
                caption = output["question"]
            
        except Exception as e:
            raise e
            
            
            return None
        
        print("*******************************************")
        print(initial_prompt.format(title = caption_title,image = image_placeHolder,related_text=related_text))
        print("+++++++++++++++++++++")
        print(refined_prompt.format(title = caption_title,image = image_placeHolder,extracted_info=extracted_info))
        print("related_text is:")
        

        print(related_text)
        print("extracted_info is: ")
        print(extracted_info)
        print("searched_image is:")
        print(img)
        print("image title (original caption title)")
        print(caption_title,keys)
        print("original caption is: ")
        print(captions)
        print("output caption is:")
        print(caption)
        if "context" in output:
            print("context is:")
            print(output["context"])
            context = output["context"]
        else:
            context = ""
        
        print("---------------------")
 
        return {
            "images": img,
            "caption": caption,
            "root": root,
            "related_text":related_text,
            "extracted_info":extracted_info,
            "original_title":captions,
            "title": caption_title,
            "context":context
        }
    # async def generate_caption(self,img,related_text,captions,root,subtitles = []):
    #     if len(subtitles) != len(img):
    #         subtitles = [""] * len(img)
    #     image_placeHolder = ""
    #     for _,subtitle in zip(img,subtitles):
    #         image_placeHolder += "<image>\n "+subtitle+"\n"
    #     initial_prompt = self.config.initial_prompt
    #     img = [os.path.join(root,image) for image in img]
    #     caption_title,keys = safe_title_from_caption_line(captions[0])
    #     # print(caption_title,keys)
    #     #  caption_title = captions[0].split(" ")[2:]
    #     # caption_title = " ".join(caption_title)
    #     try:

    #         caption = await self.vllm_response(initial_prompt.format(title = caption_title,image = image_placeHolder,related_text=related_text),img = img)
    #         caption=caption["caption"]

    #     except Exception as e:
    #         print(e)
    #         print(caption)
    #         return None
        
    #     print("*******************************************")
    #     print(initial_prompt.format(title = caption_title,image = image_placeHolder,related_text=related_text))
    #     print("related_text is:")
    #     print(related_text)
    #     print("searched_image is:")
    #     print(img)
    #     print("image title (original caption title)")
    #     print(caption_title,keys)
    #     print("original caption is: ")
    #     print(captions)
    #     print("output caption is:")
    #     print(caption)
        
    #     print("---------------------")
 
    #     return {
    #         "images": img,
    #         "caption": caption,
    #         "root": root,
    #         "related_text":related_text,
    #         "original_title":captions,
    #         "title": caption_title
    #     }


if __name__=="__main__":
    config = GeneratorConfig(core = 10,init_key="extracted" ,refined_key= "generate_prompt4.1")
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--read_path",type=str,default="/home/maxzhang/datapipeline/sub_merged_caption.json")
    parser.add_argument("--save_path",type=str,default="./temp/sub_merged_caption.json")
    # parser.add_argument("--model_name",type=str,default="ep-bi-20251126221532-kjtzx")
    parser.add_argument("--model_name",type=str,default="ep-20251028151816-lv47m")
    #parser.add_argument("--model_name",type=str,default="/mnt/storage/models/Qwen3/Qwen3_VL235B-thinking")
    
    parser.add_argument("--model_url",type=str,default="https://ark.cn-beijing.volces.com/api/v3")
    
    # parser.add_argument("--model_name",type=str,default="/mnt/storage/models/Qwen3/Qwen3-VL-thinking")
    # parser.add_argument("--model_url",type=str,default="http://0.0.0.0:8000/v1")
    
    parser.add_argument("--data_start",type=int,default=0)
    parser.add_argument("--data_end",type=int,default=100)
    parser.add_argument("--use_batch",type=bool,default=False)

    args = parser.parse_args()


    # read_path = "/mnt/storage/dataset/PPVL_reuslts_CN/中文/中文书籍/OLED显示技术_于军胜/"
    # save_path = "/mnt/storage/dataset/PPVL_reuslts_CN/storage/PROMPT12OLED显示技术_于军胜.json"
    # model_name = "/mnt/storage/models/OpenGVLab/InternVL3_5-241B-A28B"
    # model_url = "http://0.0.0.0:8081/v1"
    read_path = args.read_path
    save_path = args.save_path
    model_name = args.model_name
    model_url = args.model_url
    start = args.data_start
    end = args.data_end


    caption_generator = NewCaptionGenerator(read_path,config, save_path, model_name, model_url, api = "d69ffc82-6fdd-48ea-bff3-5dd4daf8439a",use_batch=args.use_batch)
    data_saved = []
    if os.path.exists(save_path):
        with open(save_path,"r") as f:
            data_saved = json.load(f)
    queue = Queue()
    total_num = 0
    json_data = json.load(open(read_path))
   
    total_num=len(json_data[start:end])
    total_bar = tqdm.tqdm(total = total_num)
    #allowed_tags = ["氧化物--for工艺流程-5.0","氧化物--for工艺流程-6.0","氧化物--for工艺流程-13.0"]
    def add_data(queue, read_path,start,end):
        json_data = json.load(open(read_path))
        for sample in json_data[start:end]:
            #root = sample["root"]
            root = ""
            
            img_path,captions,sbutitles,related_text,related_text_weak,tags = sample["images"],sample["figure_title"],sample["figure_title_subtitles"],sample.get("related_text_strong",None),sample.get("related_text_weak",None),sample.get("tags",None)
            # if len(img_path)!=1:
            #     continue
             
                # for ta in tags:
                #     check_tags.extend(list(ta.keys()))

                # for tag in allowed_tags:
                    
                #     if tag in check_tags:
                #         print(tag,check_tags)
           
            queue.put((captions,img_path, related_text,root,sbutitles))
               
                # # try:
                # #     related_text = search_related(text,captions[0])
                # #     # print((img_path,captions,root,sbutitles))
                #     queue.put((img_path, related_text,captions,root,sbutitles))
                # except:
                   
                #     raise ValueError(f"incorrect {img_path} {captions} {sbutitles}".format)
        for i in range(16):
            queue.put("DONE")
        print("done")
    # wait for 2 seconds
   
    workers = []
    import threading
    import multiprocessing
    import time
    process = multiprocessing.Process(target=add_data, args=(queue, read_path,start,end))
    process.start()
    tasks = []
    lock = asyncio.Lock()
    

    # wait 15 seconds for the queue to fill upq
    while True:
        if queue.qsize()>=16:
            break
        else:
            continue

    # for i in range(15):
    #     tasks.append(caption_generator.process_data(queue,bar,lock))
    print(queue.qsize())
    for i in range(16):
        tasks.append(caption_generator.process_data(queue,total_bar,lock))
    

    async def run_tasks(tasks):
        await asyncio.gather(*tasks)
        return 1
    asyncio.run(run_tasks(tasks))
    process.join()
    

    
    