RULES={}
import re
def add_rule(rule_func):
    RULES[rule_func.__name__] = rule_func
    return rule_func
def safe_title_from_caption_line(line: str) -> str:
    """
    Try to strip common 'Figure/Table X:' prefixes and return a clean title.
    """
    if not line:
        return "Untitled"
    # Remove leading Figure/Table labels like "Figure 3:", "图 2：", etc.
    #cleaned = re.sub(r'^\s*(图|图表|Figure|figure|fig|fig.)\s*\d+(?:[._-]\d+)?\s*', '', line, flags=re.I) # 图 1.1-1 iii-v led 性能的发展历程  这种情况提取出来的不对  v0.1.1
    
    #GRAPH_PREFIX_PATTERN = r'^\s*(图|图表|Figure|figure|fig|fig.)\s*\d+(?:[._-]\d+)?\s*' 图 1.1-1 iii-v led 性能的发展历程  这种情况提取出来的不对  v0.1.2添加
    #GRAPH_PREFIX_PATTERN = r'^\s*(图|图表|Figure|figure|fig|fig\.)\s*[.:-]?\s*\d+(?:[._-]\d+)*\s*[.:-]?\s*' # V0.1.5 图 2–2 求解多层膜的矩阵法的示意图  TOLED的计算机模拟及实现.md
    GRAPH_PREFIX_PATTERN =  r'^\s*(图|图表|Figure|figure|fig|fig\.)\s*[.:-]?\s*\d+(?:[._–-]\d+)*\s*[.:-]?\s*'   # V0.1.5 以后添加
    
    cleaned = re.sub(
        GRAPH_PREFIX_PATTERN,
        '',
        line,
        flags=re.I
    )

    cleaned = cleaned.strip()
    
    # Fallback: if cleaning produced empty, keep original
    if not cleaned:
        cleaned = line.strip()
   # match = re.match(r'^\s*(图|图表|Figure|figure|fig|fig.)\s*\d+(?:[._-]\d+)?', line)
    match = re.match(GRAPH_PREFIX_PATTERN, line)
    if match:
        prefix = match.group(0).strip()
        #print(prefix)
    else:
        prefix = cleaned
    return cleaned, prefix
@add_rule
def rule1(md_text,caption):
    # 规则1：查找包含“图”或“表”的文本
    related_text = []
    
    
    # for cap in caption:
    results = safe_title_from_caption_line(caption)
    _, keys = results
    # if not key1.endswith(" "):
    #     key1 += " "
    # key2 = key1.replace(" ","")
    # if not key2.endswith(" "):
    #     key2 += " "
    # key3 = " ".join(key1.split(" "))
    # key4 = key2.replace(" ","")
    keys = keys.lower()
    if keys.startswith("图"):
        key_1 =keys[1:]
        key_0 = keys[0]
    elif keys.startswith("figure"):
        key_1 =keys[6:]
        key_0 = keys[:6]
    elif keys.startswith("fig"):
        key_1 =keys[3:]
        key_0 = keys[:3]
    else:
        
        key_1 =keys
        key_0 = ""
    if " " not in keys:
        key1 = " ".join([key_0,key_1])
    else:
        key1 = keys
    key2 = key1.replace(" ","")
    
    key2_Annex = "附录"+ key2   # 附录的图跟正文图不一样
    key1_Annex = "附录" + key1  # 附录的图跟正文图不一样
    
    key1_Fu = "附" + key1       # 附图（新增）
    key2_Fu = "附" + key2       # 附图（新增）

    for line in md_text:
        
        if key1 in line.lower():
            if line.endswith(key1):  #key1刚好在在句尾的情况  v0.1.3.5之后才添加
                if "</div>" not in line.lower() and key1_Annex not in line.lower() and key1_Fu not in line.lower():
                    related_text.append(line)
            elif not line[line.index(key1)+len(key1)].isdigit():
                if "</div>" not in line.lower() and key1_Annex not in line.lower() and key1_Fu not in line.lower():
                    related_text.append(line)
        elif key2 in line:
            if line.endswith(key2):  #key1刚好在在句尾的情况  v0.1.3.5之后才添加
               if "</div>" not in line.lower() and key2_Annex not in line.lower() and key2_Fu not in line.lower():
                    related_text.append(line)
            elif not line[line.index(key2)+len(key2)].isdigit():
                if "</div>" not in line.lower() and key2_Annex not in line.lower() and key2_Fu not in line.lower():
                    related_text.append(line)

    return related_text
@add_rule
def rule2(md_text,caption):
    # 规则2 对caption 利用 jieba分词
    # 通过 N-gram 匹配每个句子中的关键词，如果在相邻上下4 个 para中找到匹配，则认为相关

    from .bm25 import BM25

    
    # remove div and "./"
    

    
    # if current line not in 

    bm25 = BM25(md_text, k1=1.5, b=0.75, use_jieba=True)
    query = caption
    top_docs = bm25.top_k(query, k=10)
    # caption id 
    start_id = None
    for doc_id, score in top_docs:
        if caption.lower() in md_text[doc_id].lower():
            start_id = doc_id
            break
    if start_id == None:
        for idx,line in enumerate(md_text):
            if caption.strip().lower() in line.lower():
                start_id = idx
                break
   
    related_texts = []
 
    for doc_id, score in top_docs:
        if score > 8:
            
            if "</div>" in md_text[doc_id] or abs(doc_id-start_id) > 2 or caption in md_text[doc_id]:
                continue
            related_texts.append(md_text[doc_id])
    return set(related_texts)
# @add_rule
# def rule2(md_text,caption):
#     # 规则2：根据 Rouge L 计算相似度
#     import jieba 
#     from jieba import posseg as pseg
#     related_text = []
#     md_text = md_text.lower()
#     md_text = md_text.split("\n")

#     # for cap in caption:
#     key_caption = caption.split(" ")
#     key1= " ".join(key_caption[:3])
#     if not key_caption[2].isdigit() and not key_caption[2].endswith("."):







def search_related(md_text,captions):
    related_text = []
    md_text = md_text.lower()
    md_text = md_text.split("\n")
    md_text = [line for line in md_text if line.strip() != ""]
   # md_text = [line for line in md_text if line.strip() != "" and (("</div>" not in line) or (captions in line))]
    final_md_text = []
    temp_stack = []
    
    for line in md_text:
        line = line.strip()
        
        if line.strip() == "":
            continue
        elif "</div>" in line:
            if captions.lower() in line:
                
                final_md_text.append(line)
            else:
                continue
        elif line.endswith("。") or line.endswith("？") or line.endswith("！") or line.endswith(".") or line.endswith("?") or line.endswith("!"):
            temp_stack.append(line)
            final_md_text.append("".join(temp_stack))
            temp_stack = []
        else:
            temp_stack.append(line)
    for rule_func in RULES:
        
        related_text.extend(RULES[rule_func](final_md_text,captions))
    return set(related_text)

def search_related_split(md_text,captions):
    related_text_SC = []
    related_text_WC_old = []
    related_text_WC = []
    md_text = md_text.lower()
    md_text = md_text.split("\n")
    md_text = [line for line in md_text if line.strip() != ""]
   # md_text = [line for line in md_text if line.strip() != "" and (("</div>" not in line) or (captions in line))]
    final_md_text = []
    temp_stack = []
    ellipsis_patterns = ("...", ".....", "……", "……", "...……","……") 
    table_pattern = r'<table.*?</table>'
    for line in md_text:
        line = line.strip()

        line = re.sub(table_pattern, '', line, flags=re.DOTALL)

        if line.strip() == "## 附录" :    #  ## 附录后面的内容都不搜索 v0.1.3添加
            break

        if line.strip() == ""  or  "keywords:" in line or "key words:" in line:  # 关键字行去掉   v0.1.3.2添加
            continue
        elif "</div>" in line:
            if captions.lower() in line:
                
                final_md_text.append(line)
            else:
                continue
        elif len(line) < 200 and any(pattern in line for pattern in ellipsis_patterns):   # 排除图像标题外的索引行 "....."的行 因为可能是插图索引   v0.1.3添加
             continue
            
        elif line.endswith("。") or line.endswith("？") or line.endswith("！") or line.endswith(".") or line.endswith("?") or line.endswith("!")  or  "....." in line or "##" in line:    # v0.1.3添加
            temp_stack.append(line)
            final_md_text.append("".join(temp_stack))
            temp_stack = []
        else:
            temp_stack.append(line)
         
    related_text_SC.extend(rule1(final_md_text,captions))
    #related_text_WC_old.extend(rule2(final_md_text, captions))
    related_text_WC = []
    for string in related_text_WC_old:
        if string not in related_text_SC:
             related_text_WC.append(string)

    return related_text_SC,related_text_WC