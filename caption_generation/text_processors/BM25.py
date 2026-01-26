# pip install jieba
import jieba
import math
from collections import Counter, defaultdict
import multiprocessing
import tqdm 


def tokenize(text):
    """
    用 jieba 把中文字符串分词，去掉空白
    你也可以在这里加停用词过滤
    """
    return [w.strip() for w in jieba.lcut(text) if w.strip()]

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, use_jieba=True):
        """
        corpus: list[str] 或 list[list[str]]
        use_jieba: True 时，对 str 自动用 jieba 分词
        """
        self.docs = []
        for doc in corpus:
            if isinstance(doc, str):
                if use_jieba:
                    tokens = tokenize(doc)
                else:
                    tokens = doc.split()
            else:
                # 已经是分好词的 list
                tokens = doc
            self.docs.append(tokens)
        
        self.N = len(self.docs)
        self.k1 = k1
        self.b = b

        # 文档长度 & 平均长度
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0

        # 构建频数和 df
        self.df = defaultdict(int)
        self.f = []
        for doc in self.docs:
            c = Counter(doc)
            self.f.append(c)
            for term in c.keys():
                self.df[term] += 1

        # 预计算 idf
        self.idf = {}
        for term, df in self.df.items():
            # 经典 BM25 idf 写法
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        self.use_jieba = use_jieba

    def _score_one(self, query_tokens, doc_id):
        score = 0.0
        doc_len = self.doc_lens[doc_id]
        freqs = self.f[doc_id]

        for term in query_tokens:
            if term not in freqs:
                continue

            tf = freqs[term]
            idf = self.idf.get(term, 0.0)

            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * tf * (self.k1 + 1) / denom

        return score

    def get_scores(self, query):
        # 查询也要用同一分词策略
        if isinstance(query, str):
            if self.use_jieba:
                query_tokens = tokenize(query)
            else:
                query_tokens = query.split()
        else:
            query_tokens = query

        scores = [self._score_one(query_tokens, i) for i in range(self.N)]
        return scores

    def top_k(self, query, k=5):
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
    def async_get_scores(self, query):
        # 查询也要用同一分词策略
        if isinstance(query, str):
            if self.use_jieba:
                query_tokens = tokenize(query)
            else:
                query_tokens = query.split()
        else:
            query_tokens = query
        parallel_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(parallel_num)
        bar = tqdm.tqdm(total=self.N)
        add_bar= tqdm.tqdm(total=self.N)
        scores = []
        for i in range(self.N):
            scores.append(self._score_one(query_tokens, i))
            bar.update(1)

        # for i in range(self.N):
    
        #     scores.append(pool.apply_async(self._score_one, args=(query_tokens, i), callback=lambda x: bar.update(1)))            
        #     add_bar.update(1)
        
       # pool.close()
        
        #pool.join()
        
        return scores


    

    def async_top_k(self, query, k=5):
        scores = self.async_get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    async def async_top_k(self, query, k=5):
        scores = await self.async_get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    # corpus = [
    #     "MicroLED 应用市场在 2023 年发生了井喷式的发展。2023 年, 芯元基在 MicroLED 微显示屏方面取得了突破进展, 攻克了像素间距  $ 7.5 \mu m $  MicroLED 芯片的阵列键合工艺, 实现了  $ 9.91 \text{ mm}(0.39 \text{ 英寸}) $  单色 MicroLED 微显示屏。JBD 开发 AIGalnP 红光 MicroLED, 突破了 100 万 nit 大关, 刷新业界记录。诺视科技报道了先键合后刻蚀的垂直堆叠技术, 实现  $ 3 \text{ mm}(0.12 \text{ 英寸}) $  MicroLED 全彩色动态图像显示, 最小像元为  $ 1.5 \mu m $ 。西安赛富乐斯发布  $ 9.91 \text{ mm}(0.39 \text{ 英寸}) $  全彩 MicroLED 显示器, 这是纳米孔量子点 (nanopores quantum dot, NPQD) 技术首次应用在全彩微显示屏领域, 标志着量子点 MicroLED 技术已成功地应用于小尺寸微显示屏。韩国名城大学发展了转移和叠层技术, 研发了 MicroLED 有源驱动阵列器件, 推动了全彩高清 MicroLED 微显示器的发展。",
    #     "这是一篇关于Python和搜索引擎的简单文档",
    #     "BM25是一种被搜索引擎广泛使用的排序函数",
    #     "影模组，可显示清晰的视频；Husion 和 LUMENS 展出了 0.8 mm 间距的 MicroLED 显示屏。2022 年，君万微发布了硅基 MicroLED 全彩微显示器，峰值亮度超 10 万 nit。"
    # ]p
    path = "/mnt/storage/dataset/PPVL_reuslts_CN/中文/中文书籍/光电显示技术及应用_文尚胜/光电显示技术及应用_文尚胜.md"


    with open(path, "r", encoding="utf-8") as f:
        corpus = f.read().split("\n")

    bm25 = BM25(corpus, k1=1.5, b=0.75, use_jieba=True)

    query = "图 8-17 消色差的投影镜头及其光学设计图"
    top_docs = bm25.top_k(query, k=10)

    for doc_id, score in top_docs:
        print(doc_id, score, corpus[doc_id])
