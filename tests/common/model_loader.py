# 从 sklearn.externals 导入 joblib
# 作用：专门用来 保存 / 读取 训练好的机器学习模型（.pkl 文件）
from sklearn.externals import joblib

def get_offline_model():
    """
    【公共工具函数】
    作用：加载本地已经训练好的离线模型，给测试用例使用
    所有指标测试、鲁棒性测试、公平性测试 都调用这一个函数拿模型
    """
    # 1. 定义模型文件路径
    # 你的模型存在项目根目录的 mlartifacts 下面，名字是 offline_model.pkl
    model_path = "./mlartifacts/offline_model.pkl"

    # 2. 用 joblib 加载模型文件
    # joblib.load() 可以把磁盘上保存的模型 读回内存，变成可使用的模型对象
    model = joblib.load(model_path)

    # 3. 把加载好的模型 返回给调用的地方（测试用例就能用了）
    return model