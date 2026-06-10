# test_ai_app_api.py
# AI应用系统接口自动化测试（带前置步骤：登录、鉴权、业务上下文）
import pytest
import mlflow
import allure
import requests
import pandas as pd

# ========== 固定 MLflow 实验 ==========
exp_name = "AI_App_System_Test"
mlflow.set_experiment(exp_name)

# ===================== 【全局前置步骤：AI应用接口基础配置】 =====================
# 这就是你要的：前置步骤 = 代码化 Setup
@pytest.fixture(scope="module", name="ai_app_headers")
def setup_ai_app_environment():
    """
    AI应用系统 全局前置步骤（一次初始化，全用例复用）
    1. 登录系统
    2. 获取token/鉴权信息
    3. 初始化接口会话
    4. 构建请求头
    """
    with allure.step("【全局前置】登录AI系统 → 获取Token"):
        # 模拟登录（真实项目替换成你们后台接口）
        login_data = {"username": "test_user", "password": "test_pass"}
        login_res = requests.post("http://your-ai-system/login", json=login_data)
        token = login_res.json().get("data", {}).get("token", "mock_token_for_test")

        # 构建通用请求头
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    # 返回给测试用例使用（全局复用）
    yield headers

    # 用例跑完后的清理动作（可选）
    with allure.step("【全局后置】退出登录/清理会话"):
        pass

# ===================== 加载测试用例数据 =====================
def get_app_test_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

test_samples = get_app_test_samples()

# ===================== AI应用接口测试用例 =====================
@allure.epic("AI应用系统测试")
@allure.feature("AI文本分类接口（业务全链路）")
@allure.story("带登录鉴权·系统可用性+正确性验证")
@allure.title("AI系统接口测试：{raw_data[input_text]}")
@pytest.mark.ai_app_api
@pytest.mark.parametrize("raw_data", test_samples)
def test_ai_app_classify_api(ai_app_headers, raw_data):
    """
    AI应用系统接口自动化测试：
    1. 已自动完成：登录、token、上下文
    2. 调用真实AI接口
    3. 校验：接口状态 + 业务结果 + AI预测正确性
    """
    # 测试数据
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]
    ai_api_url = "http://your-ai-system/api/classify"

    # MLflow 记录
    with mlflow.start_run(nested=True, run_name=f"AI系统接口_{text[:18]}"):
        mlflow.log_param("test_type", "ai_app_api_system")
        mlflow.log_param("input_text", text[:30])
        mlflow.log_param("true_label", true_label)

        # ===================== 业务测试步骤 =====================
        with allure.step("步骤1：组装AI接口请求参数"):
            request_body = {
                "content": text,
                "scene": "comment_classify",
                "version": "v1.0"
            }

        with allure.step("步骤2：调用AI应用接口（带鉴权）"):
            response = requests.post(
                url=ai_api_url,
                headers=ai_app_headers,
                json=request_body,
                timeout=10
            )

        with allure.step("步骤3：校验接口可用性（状态码=200）"):
            assert response.status_code == 200, f"接口异常，状态码：{response.status_code}"
            res_data = response.json()
            mlflow.log_param("response_code", res_data.get("code", -1))

        with allure.step("步骤4：校验AI业务结果正确性"):
            predict_label = int(res_data["data"]["label"])
            predict_score = round(float(res_data["data"]["score"]), 4)

            # 记录指标
            mlflow.log_metric("predict_label", predict_label)
            mlflow.log_metric("predict_score", predict_score)

        with allure.step("步骤5：校验模型预测准确性"):
            is_correct = 1 if predict_label == true_label else 0
            mlflow.log_metric("is_correct", is_correct)

        # ===================== 质量门禁 =====================
        assert is_correct == 1, f"预测错误！真实标签={true_label}, 预测={predict_label}"
        assert predict_score >= 0.5, f"置信度过低：{predict_score}"