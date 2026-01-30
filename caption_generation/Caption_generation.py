
import json 
import os 
import tqdm 
import asyncio
from multiprocessing import Queue
import openai 
from openai import AsyncClient
import base64

from Prompts import Prompt
from text_processors import safe_title_from_caption_line, search_related
from pathlib import Path
import re
import markdown
import json
import os
import json 

import re
from typing import Optional, Tuple
FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*|\s*```$", re.S)
def _strip_code_fences(s: str) -> str:
    return FENCE_RE.sub("", s.strip())

def _find_json_span(s: str) -> Optional[Tuple[int, int]]:
    # 在整段文本中找到第一个完整的 {...} 或 [...] 片段（支持字符串与转义）
    opens = ["{", "["]
    closes = {"{": "}", "[": "]"}
    i = min((s.find(ch) for ch in opens if s.find(ch) != -1), default=-1)
    if i == -1:
        return None
    stack = [s[i]]
    in_str = False
    esc = False
    j = i + 1
    while j < len(s):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch in opens:
                stack.append(ch)
            elif ch in closes.values():
                if not stack:
                    return None
                top = stack.pop()
                if ch != closes[top]:
                    return None
                if not stack:
                    return (i, j + 1)
        j += 1
    return None

def _basic_normalize(s: str) -> str:
    # 常见“类 JSON”修复：去拖尾逗号、替换 Python 字面量、统一智能引号
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # 去掉对象/数组结尾处多余逗号
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # 将未加引号的键补上引号（只在 { ... } 范围内粗略处理）
    def quote_keys(m):
        before, key, after = m.groups()
        return f'{before}"{key}"{after}:'
    s = re.sub(r'(?P<pre>[{,\s])(?P<key>[A-Za-z_][\w\-]*)\s*:', lambda m: quote_keys((m.group('pre'), m.group('key'), "")), s)
    # Python -> JSON 布尔/空值
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s

def _last_resort_quotes(s: str) -> str:
    # 兜底：将单引号字符串替换为双引号（可能伤害包含撇号的英文，故放在最后一步）
    # 仅替换形如 '...'(在字符串外部) 的内容
    def repl(m):
        inner = m.group(1).replace('\\"', '"').replace('"', '\\"')
        return f'"{inner}"'
    s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", repl, s)
    return s

def extract_json(response: str):
    """
    尝试从大模型输出中提取并解析 JSON（dict/list）。
    优先原生 JSON，其次 Python 字面量，最后正则修复。
    失败会抛出 ValueError 并附带上下文。
    """
    raw = response.strip()
    # 情况 A: 整段就是 JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 去除 markdown 代码围栏
    nofence = _strip_code_fences(raw)

    # 再试一次直接 JSON
    try:
        return json.loads(nofence)
    except Exception:
        pass

    # 从文本中定位一个 JSON 片段
    span = _find_json_span(nofence)
    candidate = nofence if span is None else nofence[span[0]:span[1]]

    # 直接 JSON 解析
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        return eval(candidate)
    except Exception:
        pass

    # 尝试 Python 字面量（支持单引号、None/True/False）
    try:
        obj = ast.literal_eval(candidate)
        # 确保是可 JSON 化的
        json.dumps(obj)
        return obj
    except Exception:
        pass

    # 进行正则级修复
    fixed = _basic_normalize(candidate)

    # 再试 JSON
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # 兜底：强行把单引号串替换为双引号
    fixed2 = _last_resort_quotes(fixed)

    try:
        return json.loads(fixed2)
    except Exception as e:
        ctx = (candidate if len(candidate) < 800 else candidate[:800] + "...<truncated>")
        raise ValueError(f"Failed to parse JSON after normalization. Last error: {e}\nCandidate snippet:\n{ctx}")

def read_md_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                    yield content,root

def parse_by_render(md_text: str):
    '''
    find the corresponding image and caption in the markdown file:
    first, we believe that if there is only an image and a caption surrounded by texts,
    then the image and caption are corresponding.
    however, if there consecutive images and captions, then we should judge the corresponding and relations by texts
    if a caption is started with "figure, fig, table,图 表", then it is a set of caption and image
    if there are multiple image and only one caption,  then we should merge the images together and join all the captions
    if there are multiple group of caption and image, we should link image and caption according to the order (each caption join images)
    # elif len(stacks_text)==0 and len(stacks_images)>0:
    #     final_json["images"].append(stacks_images)
    #     stacks_images = []
    
    '''
    stacks_text = []
    stacks_images= []
    stacked_subtitles = []
    final_json = {
                "images":[],
                "captions": [],
                "subtitles": [],
                }
    i=0
    md_text = md_text.split("\n")
    md_text = [x for x in md_text if x.strip() != ""]
    while i < len(md_text):
        line = md_text[i].strip()
        if line.strip() == "## 附录":    #  ## 附录后面的图都不考虑  v0.1.2
            break

        if "div style=\"text-align: center;\"" not in line:
            if len(stacks_images)==0 and len(final_json["images"])-len(final_json["captions"])==1:
                if len(stacks_text) ==0:
                    raise ValueError("there is no caption for the image")
                final_json["captions"].append(stacks_text)
                stacks_text = []
            elif len(stacks_images)==0 and len(final_json["images"])-len(final_json["captions"])==-1:
                if len(stacks_text) ==0:
                    raise ValueError("there is no caption for the image")
                final_json["captions"].append(stacks_text)
                stacks_text = []
            i+=1
            stacks_text = []
            stacks_images= []
            stacked_subtitles = []
            
           
        else:
            img_flag = 0
            while i < len(md_text):
                line = md_text[i].strip()
                #print(line)    
                # parse the div element 
                if "<img" in line and "div style=\"text-align: center;\"" in line:
                    image_path = re.search(r'src="([^"]+)"', line).group(1)
                    stacks_images.append(image_path)
                    img_flag = 1
                    
                        
                elif "div style=\"text-align: center;\"" in line:
                    text = re.search(r'>(.*?)</div', line).group(1)
                    text = text.replace("\n", "")
                    text = text.lower()
                    # print(text)
                    # print(stacks_images)
                
                    if text.startswith("figure") or text.startswith("fig") or text.startswith("图"):
                        stacks_text.append(text)    
                        if stacks_images != []:
                            final_json["images"].append(stacks_images.copy())
                            # if len(stacked_subtitles) == len(stacks_images):
                            #     final_json["subtitles"].append(stacked_subtitles.copy())
                            # else:
                            #     final_json["subtitles"].append([""])
                            final_json["subtitles"].append(stacked_subtitles.copy())
                            final_json["captions"].append(stacks_text)
                            stacked_subtitles = []
                            stacks_text=[]
                            stacks_images=[]
                        elif len(final_json["images"]) == len(final_json["captions"]) and len(final_json["captions"]) >0:
                            if img_flag:
                                final_json["captions"][-1] += stacks_text
                    
                            stacks_text=[]
                    elif not text.startswith("table") and not text.startswith("表"):
                        stacked_subtitles.append(text)

                else:
                    break
                i+=1


            


    return final_json
import torch
import gc
from volcenginesdkarkruntime import AsyncArk


class GeneratorConfig:
    def __init__(self, **Kwargs):
        init_key = Kwargs['init_key']
        refined_key = Kwargs['refined_key']
        self.initial_prompt = Prompt[init_key]
        #self.final_prompt = Prompt['final_prompt']
        self.refined_prompt = Prompt[refined_key]     
        self.core = Kwargs['core']


class CaptionGenerator:
    def __init__(self, read_path,config, save_path, model_name,model_url,generator_saved="/mnt/storage/dataset/PPVL_reuslts_CN/storage/generator.json", **Kwargs):
        self.read_path = read_path
        self.save_path = save_path
        self.model_name = model_name
        self.model_url = model_url
        self._data = []
        self.use_batch = Kwargs.get("use_batch",False)
        if not self.use_batch:
            self._client = AsyncClient(
            base_url=self.model_url,
            api_key=Kwargs.get("api","")
            )
        else:
            self._client = AsyncArk(
               
                api_key=Kwargs.get("api",""),
                timeout=24 * 3600,
            )
        
        self.generator_saved = generator_saved
        self.config = config
        use_offline = Kwargs.get("use_offline",False)
        self.use_offline =use_offline
        if use_offline:
            self.deploy_offline()
    def deploy_offline(self):
        
        engine_config = PytorchEngineConfig(tp=8,session_len=8192,cache_max_entry_count=0.9, enable_mp_engine=True)
        self.pipe = pipeline(self.model_name,
                        backend_config=engine_config)
    def reboot_pipeline(self):
        self.pipe.close()
        self.pipe = None
        self.deploy_offline()
        torch.cuda.empty_cache()
        gc.collect()
    def read_data(self):
        #
        for root, dirs, files in os.walk(self.read_path):
            
            parent_path = os.path.dirname(root)
            for file in files:

                if file.endswith(".md"):
                    self._data.append((root, os.path.join(root, file)))
                    break
                # self._data.append(os.path.join(root, file))
                # search for markdown file in 
        #print(self._data)
        return self._data
    @staticmethod
    def recall_caption(markdown_path):
        # read markdown file 
        with open(markdown_path, "r") as f:
            markdown_data = f.read()
        text_json = parse_by_render(markdown_data)
        if len(text_json["images"]) == len(text_json["captions"]):
            text_json["text"] = markdown_data
            return text_json
        else:
            return None
    def progress_bar(self):
        self.read_data()
        total_num = 0
        for root, md_path in self._data:
            text_json = self.recall_caption(md_path)
            total_num += len(text_json["images"])
        bar = tqdm.tqdm(total=total_num)
        return bar
    def get_related_text(self,images,captions,subtitles):
        return search_related(text_json["text"],captions[0])
    def caption_GENERATOR(self,save = True):
        self.read_data()
        
        # if os.path.exists(self.generator_saved):
        #     scanned_mds = json.load(open(self.generator_saved))
        # else:
        scanned_mds = {}
        for root, md_path in self._data:
            if md_path in scanned_mds:
                for images,related_text,captions,root,subtitles in scanned_mds[md_path]:
                    yield images,related_text,captions,root,subtitles
            else:
                text_json = self.recall_caption(md_path)
                md_path_data = []
                if text_json is None:
                    continue
                for images,captions,subtitles in zip(text_json["images"],text_json["captions"],text_json["subtitles"]):
                    try:
                        related_text = search_related(text_json["text"],captions[0])
                    
                        md_path_data.append((images,related_text,captions,root,subtitles)) 
                        yield images,related_text,captions,root,subtitles
                    except:
                        continue
                scanned_mds[md_path] = md_path_data
                # if save:
                #     with open(self.generator_saved, 'w',encoding = "utf-8") as f:
                #         json.dump(scanned_mds, f, ensure_ascii=False, indent=4)

    async def vllm_response(self, prompt,img = None):
        if img is None:
            message = [
                {"role": "user", "content": prompt},
            ]
        else:
            
            message = [
                {"role": "system", "content": [
                    {
                        "type": "image_url", "image_url":{
                            "url": self.encode_image(image)
                        } 
                    } for image in img] + \
                    [{
                        "type": "text", "text":  prompt
                        
                    }]
                },
                 {"role": "user", "content": [
                    {
                        "type": "text", "text":  "输出的json 为:"
                        
                    }]
                }
            ]
        
        try:
            if self.use_offline:
            
                response = self.pipe(message,max_new_tokens = 8192)
               
                
                response = response.text
            #print(prompt,img)
            else:
               
                if not self.use_batch:
                    print("send one")
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        max_tokens=8192,
                    )
                else:
                    print("send one batch")
                    response = await self._client.batch.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        max_tokens=8192,
                    )
                print("receive one")
                response = response.choices[0].message.content
               

        except Exception as e:
            print("connection error", e)

            # print(prompt,img)
            # print(response)
            import subprocess
            
            # if "finished, reason \"error\"" in str(e) and not self.use_offline:
            #     print("relaunch server")
            #     # 同步执行两个命令：先 pkill 再启动服务
            #     cmd = "pkill -f lmdeploy"
            #     subprocess.run(cmd, shell=True, check=True)  # 阻塞等待命令执行完成
            # self.use_offline = True
            # self.reboot_pipeline()
            return None
        response = self.parse_json(response)
        return response
    async def generate_caption(self,img,related_text,captions,root,subtitles = []):
        if len(subtitles) != len(img):
            subtitles = [""] * len(img)
        image_placeHolder = ""

        image_placeHolder += "<image>\n"
        initial_prompt = self.config.initial_prompt
        caption_title,keys = safe_title_from_caption_line(captions[0])
        caption_title += "\n"+ "\n".join(subtitles)
      
        
        caption = await self.vllm_response(initial_prompt.format(title = caption_title,image = image_placeHolder))
    
        if caption == None:
            return None
        caption = caption["caption"]
        related_text = "\n\n".join(related_text)

        if len(related_text.strip()) > 0:
            
            refined_prompt = self.config.refined_prompt.format(related_text=related_text,title = caption_title, original_caption = caption,image = image_placeHolder)
           
            refined_caption = await self.vllm_response(refined_prompt,img)
            if refined_caption == None:
                return None
            refined_caption = refined_caption["refined_caption"]
            print("+++++++++++++++++++++++++++++++++++")
            print(refined_prompt)
        else:
            refined_caption = caption

        
        print("*******************************************")
        print(initial_prompt.format(title = caption_title,image = image_placeHolder))
        
        print("related_text is:")
        print(related_text)
        print("searched_image is:")
        print(img)
        print("image title (original caption title)")
        print(caption_title)
        print("refined_caption is: ")
        print(refined_caption)
        print("original caption is: ")
        print(caption)
        print(captions)
        print("---------------------")
 
        return {
            "images": img,
            "caption": refined_caption,
            "original_caption": caption,
            "root": root,
            "related_text":related_text,
            "original_title":captions,
            "title": caption_title
        }

        # filter_caption_score = self.check_caption(img,refined_caption)
        # if filter_caption_score> self.threshold:
        #     return img,filter_caption["caption"]
        # else:
        #     return None
    def encode_image(self,image):

        image_type = image.split('.')[-1]
        encoded_image  = base64.b64encode(open(image, 'rb').read()).decode('utf-8')
        return f'data:image/{image_type};base64,{encoded_image}'

    #return cleaned or line.strip()

    @staticmethod
    def parse_json(text):
        
        
        
        return extract_json(text.strip())
    def filter_caption(img,related_text,caption):
        prompt_filter = self.get_prompt(self.config.filter_prompt_index)
        scores = []
        for i in range(self.config.filter_prompt_num):
            response = self.vllm_response(prompt_filter.format(caption = caption,related_text = related_text),img)
            if response == None:
                continue
            score = response["score"]
            scores.append(score)
        filter_cpation_score = sum(scores)/(len(score)+0.000000000000001)
        return filter_cpation_score
    def deploy_llm(self,llm_path):
        pass


    async def process_data(self,input_queue,bar,load_lock = None):
        existed_captions = []
        if os.path.exists(self.save_path):
            with open(self.save_path,'r') as f:
                data = json.load(f)
            for sample in data:
                existed_captions.extend(sample["figure_title"])
        print("start process")
        while True:
            # if input_queue.empty():
            
            #     continue
            result = input_queue.get()
           
            if result =="DONE":
                break
            else:
                caption = result[0]

            if caption[0] in existed_captions:
                bar.update(1)
                continue
                # print("exitsted",caption[0],self.save_path)
                
            print("generate one")
           
            data = await self.generate_caption(result)
            
            
            if data != None:
                async with load_lock:
                    if os.path.exists(self.save_path):
                        with open(self.save_path,"r") as f:
                            history_data = json.load(f)
                    else:
                        os.makedirs(os.path.dirname(self.save_path),exist_ok=True)
                        history_data = []
                    history_data.append(data)
                
                    with open(self.save_path,"w") as f:
                        json.dump(history_data,f,ensure_ascii=False,indent=4)
            else:
                input_queue.put(result)
                continue

           
            
            bar.update(1)
        print("finished")
           

if __name__ =="__main__":   
    config = GeneratorConfig(core = 10,init_key="extracted" ,refined_key= "generate_prompt4")
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--read_path",type=str,default="/mnt/storage/dataset/PPVL_reuslts_CN/中文/中文书籍/OLED显示技术_于军胜/")
    parser.add_argument("--save_path",type=str,default="/mnt/storage/dataset/PPVL_reuslts_CN/storage/PROMPT12OLED显示技术_于军胜.json")
    parser.add_argument("--model_name",type=str,default="ep-20251124134322-8h8rn")
    parser.add_argument("--model_url",type=str,default="https://ark.cn-beijing.volces.com/api/v3")


    args = parser.parse_args()


    # read_path = "/mnt/storage/dataset/PPVL_reuslts_CN/中文/中文书籍/OLED显示技术_于军胜/"
    # save_path = "/mnt/storage/dataset/PPVL_reuslts_CN/storage/PROMPT12OLED显示技术_于军胜.json"
    # model_name = "/mnt/storage/models/OpenGVLab/InternVL3_5-241B-A28B"
    # model_url = "http://0.0.0.0:8081/v1"
    read_path = args.read_path
    save_path = args.save_path
    model_name = args.model_name
    model_url = args.model_url


    caption_generator = CaptionGenerator(read_path,config, save_path, model_name, model_url,api = "40e2b23b-f89a-416f-bb87-d32bf448dc20")
    data_saved = []
    if os.path.exists(save_path):
        with open(save_path,"r") as f:
            data_saved = json.load(f)

    caption_generator.read_data()
    tasks = []

    bar = caption_generator.progress_bar()
    queue = Queue()
    def add_data(queue, caption_generator):
        for img_path, related_text,caption,root,sbutitles in caption_generator.caption_GENERATOR():
            # print(img_path,related_text,caption)
            #print(len(related_text))
            queue.put((img_path, related_text,caption,root,sbutitles))
        queue.put("DONE")
    # wait for 2 seconds
   
    workers = []
    import threading
    import multiprocessing
    import time
    process = multiprocessing.Process(target=add_data, args=(queue, caption_generator))
    process.start()
    tasks = []
    lock = asyncio.Lock()
    
    # wait 15 seconds for the queue to fill upq
    # while True:
    #     if queue.qsize()>=15:
    #         break
    #     else:
    #         continue
    process.join()
    # for i in range(15):
    #     tasks.append(caption_generator.process_data(queue,bar,lock))
    for i in range(10):
        tasks.append(caption_generator.process_data(queue,bar,lock))
    async def run_tasks(tasks):
        await asyncio.gather(*tasks)
    asyncio.run(run_tasks(tasks))

    
    # 

    
    
            


    
        