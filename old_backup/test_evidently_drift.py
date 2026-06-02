import pytest
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import os
import allure

# 补充Allure用例注解（优化报告展示）
@pytest.mark.drift
@allure.feature("AI模型监控")
@allure.story("数据漂移与模型质量检测")
@allure.title("数据漂移+模型质量综合测试")
def test_evidently_data_drift_and_model_quality():
    """【AI模型监控】数据漂移 + 模型质量测试（可在Jenkins运行）"""

    # ===================== 1. 构造数据 =====================
    # 👉 Evidently 分类评估要求标签列固定为 target
    reference_data = pd.DataFrame({
        "text": ["good movie", "bad movie", "nice story", "boring plot"] * 50,
        "target": [1, 0, 1, 0] * 50
    })
    current_data = pd.DataFrame({
        "text": ["awesome film", "awful acting", "great plot", "bad script"] * 50,
        "target": [1, 0, 1, 0] * 50
    })

    # ===================== 2. 训练模型 =====================
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(reference_data["text"], reference_data["target"])

    # ===================== 3. 生成预测 =====================
    reference_data["prediction"] = model.predict(reference_data["text"])
    current_data["prediction"] = model.predict(current_data["text"])

    # ===================== 4. 生成 数据漂移报告 =====================
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 优化：使用相对路径，兼容多环境（Jenkins/本地）
    report_dir = "./evidently_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    drift_path = os.path.join(report_dir, "data_drift_report.html")
    data_drift_report.save_html(drift_path)

    # ===================== 5. 生成 模型质量报告 =====================
    model_quality_report = Report(metrics=[ClassificationPreset()])
    model_quality_report.run(reference_data=reference_data, current_data=current_data)
    
    quality_path = os.path.join(report_dir, "model_quality_report.html")
    model_quality_report.save_html(quality_path)

    # ===================== 6. 【核心修复】正确附加HTML文件到Allure =====================
    # 方式1：官方推荐 attach.file（优先使用，专门处理本地文件）
    allure.attach.file(
        source=drift_path,
        name="数据漂移报告",
        attachment_type=allure.attachment_type.HTML
    )
    allure.attach.file(
        source=quality_path,
        name="模型质量报告",
        attachment_type=allure.attachment_type.HTML
    )

    # 备选方式（保留原写法，修复编码+句柄问题，二选一即可）
    # with open(drift_path, 'r', encoding='utf-8') as f:
    #     allure.attach(body=f.read(), name="数据漂移报告", attachment_type=allure.attachment_type.HTML)
    # with open(quality_path, 'r', encoding='utf-8') as f:
    #     allure.attach(body=f.read(), name="模型质量报告", attachment_type=allure.attachment_type.HTML)

    # ===================== 7. 断言（CI/CD 门禁） =====================
    assert os.path.exists(drift_path), "数据漂移报告生成失败"
    assert os.path.exists(quality_path), "模型质量报告生成失败"
    assert not reference_data["prediction"].isnull().any(), "参考数据预测失败"
    assert not current_data["prediction"].isnull().any(), "当前数据预测失败"

    print("✅ 【测试通过】Evidently 报告已生成完成！")