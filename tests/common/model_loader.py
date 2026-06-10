#========== 原有sklearn joblib加载代码完整保留 ==========
# 从 sklearn.externals 导入 joblib
# 作用: 专门用来 保存 / 读取 训练好的机器学习模型 (.pkl 文件)
try:
    from sklearn.externals import joblib
except:
    import joblib

# ========== 新增transformers、torch 依赖（原有BERT代码） ==========
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os

# 本地BERT模型根目录（截图里的离线路径）
BASE_MODEL_PATH = "/home/ubuntu/Desktop/ai-models"

# ---------------- BERT系列加载函数（保留） ----------------
def get_distilbert_chinese():
    """加载中文distilbert-base-chinese预训练模型（文本分类底座）"""
    model_dir = os.path.join(BASE_MODEL_PATH, "distilbert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval() # 推理模式，关闭dropout
    return model, tokenizer

def get_all_minilm():
    """加载all-MiniLM-L6-v2英文向量化模型（语义相似度用）"""
    model_dir = os.path.join(BASE_MODEL_PATH, "all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return model, tokenizer

# 封装文本预测工具（适配原有测试用例的predict接口）
class BertOfflineModel:
    def __init__(self):
        self.bert_model, self.tokenizer = get_distilbert_chinese()
        # 可在这里加载训练好的分类头权重（后续微调后补充）
    
    def predict(self, text_list):
        """兼容原有测试代码 model.predict([text]) 调用格式"""
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.bert_model(**inputs)
        # 临时模拟分类输出（后续微调训练后替换真实分类逻辑）
        return [1 if len(t) > 3 else 0 for t in text_list]

# ---------------- 【原版函数完整保留，二选一使用】 ----------------
def get_offline_model():
    """
    【公共工具函数】
    作用: 加载本地已经训练好的离线模型, 给测试用例使用
    所有指标测试、鲁棒性测试、公平性测试 都调用这一个函数拿模型

    切换方案：注释其中一段即可切换sklearn/pkl 或 DistilBERT模型
    """
    # ========== 方案1：原始sklearn pkl模型（图片中原代码，默认注释，需要就取消注释） ==========
    """
    # 1. 定义模型文件路径
    # 你的模型存在项目根目录的 mlartifacts 下面, 名字是 offline_model.pkl
    model_path = "./mlartifacts/offline_model.pkl"
    # 2. 用 joblib 加载模型文件
    # joblib.load() 可以把磁盘上保存的模型 读回内存, 变成可使用的模型对象
    model = joblib.load(model_path)
    # 3. 把加载好的模型 返回给调用的地方 (测试用例就能用了)
    return model
    """

    # ========== 方案2：DistilBERT中文离线大模型（当前启用） ==========
    return BertOfflineModel()

# 额外预留：单独获取MiniLM向量模型（用于drift数据漂移用例）
def get_minilm_model():
    return get_all_minilm()