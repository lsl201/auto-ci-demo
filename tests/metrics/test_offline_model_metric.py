import allure
# ==========1.导入依赖包==========
import pytest
# pytest测试框架核心库，实现自动化用例执行、标记、断言
import pandas as pd
# 表格数据处理，读取csv离线测试数据集
from sklearn.metrics import accuracy_score, confusion_matrix
# 机器学习指标：accuracy=准确率，confusion_matrix=混淆矩阵（拆解四类误判）
import mlflow
# MLflow实验追踪：记录参数、指标，留存模型测试数据，版本对比
from tests.common.model_loader import get_offline_model
# 导入自定义公共函数：从项目common目录加载离线保存好的模型文件

# ==========2.全局常量：MLflow实验配置==========
EXPERIMENT = "AI_Security_Fairness_Test"
# 定义实验名，所有离线指标测试统一归入该MLflow实验，方便UI集中查看
mlflow.set_tracking_uri("file:./mlruns")
# 配置MLflow存储路径：指标、参数落地保存到本地./mlruns文件夹，离线可用，不依赖远端服务
mlflow.set_experiment(EXPERIMENT)
# 将后续测试Run绑定上面定义的实验

# ==========3.用例装饰器（Pytest+Allure报告配置）==========
@pytest.mark.metric
# pytest自定义标记metric，执行命令`pytest -m metric`可只筛选运行本模块所有指标用例，适配CI流水线分组执行
@allure.feature("离线模型业务指标验收")
# Allure报告大模块分类：报告左侧菜单【离线模型业务指标验收】，区分功能模块
@allure.title("离线全量测试集：准确率+误判率核算")
# Allure单条用例标题，生成测试报告时展示用例名称，替代默认函数名，可读性更强
def test_offline_acc_err_rate():
    """离线模型全量数据集测试：计算准确率、整体误判率、FPR假正误判、FNR假负误判"""
    # ==========4.MLflow上下文：开启单次测试Run（一次测试=一条Run记录）==========
    with mlflow.start_run(run_name="离线模型指标验收_准确率误判率"):
    # with自动管理run生命周期：代码块开始创建Run，代码结束自动关闭Run，无需手动end_run；run_name自定义本次测试名称

        # 4.1读取离线固定标注测试集
        df = pd.read_csv("tests/data/offline_test_dataset.csv")
        # 读取项目固定离线测试csv：固定数据集=基准，每次测试复用同一数据，保证指标变化只来自模型，不受测试集干扰
        x_data = df["input_text"].tolist()
        # 提取输入特征（模型入参文本）转为列表
        y_true = df["true_label"].tolist()
        # 提取真实标注标签（人工标准答案0/1）
        mlflow.log_param("test_sample_num", len(df))
        # log_param记录**参数（静态值）**：把测试样本总数存入MLflow，后续查看报告可知本次测试样本量

        # 4.2加载离线模型
        model = get_offline_model()
        # 调用公共方法，读取项目目录离线保存的pkl模型文件，完成模型加载（不用重复训练，纯离线验收）
        y_pred = model.predict(x_data)
        # 全量测试集批量推理，生成模型预测标签

        # 4.3核心指标计算（准确率+多层误判率）
        acc = accuracy_score(y_true, y_pred)
        # 准确率=预测正确样本/总样本，业务核心指标
        err_total = 1 - acc
        # 整体误判率=1-准确率，全样本错误占比
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # confusion_matrix生成2*2混淆矩阵；ravel()扁平化展开成4个值
        # TN：真负(负样本预测正确)、FP：假正(负错判正→误杀)、FN：假负(正错判负→漏放)、TP：真正(正样本预测正确)
        fpr = fp/(fp+tn) if (fp+tn)!=0 else 0
        # FPR假正误判率：负样本里被错误判成正的比例（业务误拦截率），分母防除0报错
        fnr = fn/(fn+tp) if (fn+tp)!=0 else 0
        # FNR假负误判率：正样本里被错误判成负的比例（业务漏检率）

        # 4.4MLflow持久化指标（数值类用log_metric）
        mlflow.log_metric("offline_accuracy", acc)
        # 记录离线准确率，MLflow保存历史数值，迭代新版本可对比指标涨跌
        mlflow.log_metric("total_error_rate", err_total)
        # 记录整体误判率
        mlflow.log_metric("false_positive_rate_FPR", fpr)
        # 记录误杀率FPR
        mlflow.log_metric("false_negative_rate_FNR", fnr)
        # 记录漏放率FNR

        # 4.5业务门禁断言：不达标直接用例失败、阻断CI上线
        assert acc >= 0.90, f"离线准确率{acc:.2%} < 90%，指标不达标"
        # 业务阈值：准确率低于90%抛出异常，pytest标记用例失败
        assert err_total <= 0.10, f"整体误判率{err_total:.2%} >10%，上线风险超标"
        # 整体错误率超10%阻断上线，是AI模型上线质量门禁