import numpy as np
from scipy.spatial.distance import cosine
import zh_core_web_lg
from transformers import AutoModel, AutoTokenizer

# 加载预训练模型
en_nlp = en_core_web_lg.load()
zh_nlp = zh_core_web_lg.load()
vi_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
vi_model = AutoModel.from_pretrained("vinai/phobert-base")

# 获取单词向量函数
def get_word_vector(lang, word):
    if lang == "en":
        return en_nlp(word).vector
    elif lang == "zh":
        return zh_nlp(word).vector
    elif lang == "vi":
        inputs = vi_tokenizer(word, return_tensors="pt")
        outputs = vi_model(​**​inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# 跨语言映射矩阵（示例使用MUSE预训练矩阵）
def load_mapping_matrix(src_lang, tgt_lang="en"):
     # 文件路径配置（需根据实际路径修改）
    muse_path = {
        "zh": "pretrained/muse/wiki.multi.zh.vec",
        "vi": "pretrained/muse/wiki.multi.vi.vec"
    }[src_lang]
    
    # 加载二进制格式的映射矩阵（更快更可靠）
    if not Path(muse_path).exists():
        raise FileNotFoundError(f"请先从MUSE官网下载{src_lang}-{tgt_lang}的映射矩阵")
    
    # 读取300x300的映射矩阵
    with open(muse_path, 'rb') as f:
        # 跳过前2字节的UTF-8 BOM（针对某些文本格式矩阵）
        if src_lang == "zh":
            f.seek(2)
        # 读取300x300 float32矩阵
        M = np.fromfile(f, dtype=np.float32, count=300 * 300)
        return M.reshape(300, 300).T  # 转置为标准投影矩阵
    # 此处需要您下载MUSE的预训练矩阵
    # 例如中文到英文：https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.zh.vec
    # 越南语到英文：https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.vi.vec
    # 返回形状为 (300, 300) 的numpy矩阵
    

# 计算跨语言相似度
def crosslingual_similarity(src_word, src_lang, tgt_word, tgt_lang="en"):
    # 获取源向量
    src_vec = get_word_vector(src_lang, src_word)
    
    # 加载映射矩阵
    M = load_mapping_matrix(src_lang) 
    
    # 映射到英文空间
    mapped_vec = np.dot(src_vec, M)
    
    # 获取目标词向量
    tgt_vec = get_word_vector(tgt_lang, tgt_word)
    
    # 计算余弦相似度
    return 1 - cosine(mapped_vec, tgt_vec)

# 中文词到英文的相似度
print(crosslingual_similarity("苹果", "zh", "apple"))  # 预期输出约0.75

# 越南语到英文的相似度 
print(crosslingual_similarity("công_ty", "vi", "company"))  # 预期输出约0.68