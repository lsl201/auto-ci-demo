# online_api_server.py 本地模拟线上HTTP服务，地址：http://127.0.0.1:8000/predict
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 加载训练好的模拟线上小模型
model = joblib.load("./online_sim_model.joblib")
app = FastAPI(title="模拟线上AI推理服务")

# 入参格式：批量文本列表，和原有调用逻辑匹配
class ReqData(BaseModel):
    text_list: list[str]

# 推理接口（对标第三方API的/v1/chat/completions，但是本地闭环）
@app.post("/predict")
def predict(req: ReqData):
    pred = model.predict(req.text_list).tolist() # 返回[0,1,0...]和离线模型输出格式一致
    return {"pred_label": pred}

# 启动命令：uvicorn online_api_server:app --host 127.0.0.1 --port 8000