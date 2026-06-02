# -*- coding: utf-8 -*-
import allure
import pytest
import warnings
import random
import string
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

# ====================== 全局配置（和原有文件保持一致） ======================
warnings.filterwarnings("ignore")

# ------------ 请根据你的项目替换以下模型/数据 ------------
# 模拟加载你的训练好的模型（替换为你真实模型对象）
class MockModel:
    def predict(self, texts):
        """模拟文本分类预测：0=负面，1=正面"""
        res = []
        for t in texts:
            # 正常文本简单判断
            if any(w in t.lower() for w in ["good", "great", "nice"]):
                res.append(1)
            elif any(w in t.lower() for w in ["bad", "terrible", "worst"]):
                res.append(0)
            else:
                # 脏输入默认返回0，模拟模型兜底
                res.append(0)
        return np.array(res)

# 全局模型实例（替换为你真实模型）
model = MockModel()

# 公平性审计模拟数据（敏感特征：年龄分组、性别，按需替换）
fairness_data = pd.DataFrame({
    "text": [
        "This movie is good", "I hate this film", "Nice story", "Too bad",
        "Great acting", "Worst ever", "Fun watch", "Boring"
    ] * 10,
    "label": [1, 0, 1, 0, 1, 0, 1, 0] * 10,
    "age_group": ["young", "old"] * 40,   # 敏感属性1
    "gender": ["male", "female"] * 40     # 敏感属性2
})
# =============================================================================

# ====================== 工具函数（内部调用，无需修改） ======================
def generate_random_fuzz_str(length: int = 50):
    """生成模糊测试脏字符串：字母、数字、特殊符号、emoji、乱码"""
    chars = string.printable + "￥%……&*（）<>/\\'\"`~" + "😡🤯💀"
    return ''.join(random.choice(chars) for _ in range(length))


def get_dirty_input_cases():
    """构造全量脏输入用例集"""
    cases = [
        # 1. 空值类
        ("empty_str", "", "空字符串"),
        ("all_whitespace", "   \t\n\r", "全空白符"),
        ("none_input", None, "None空对象"),
        # 2. 超长文本
        ("very_long_text", "good " * 1000, "超长合法文本"),
        ("super_long_dirty", generate_random_fuzz_str(2000), "超长脏字符串"),
        # 3. 特殊字符 & 脚本注入
        ("special_chars", "!@#$%^&*()_+-=[]{}|;':\",./<>?", "全特殊符号"),
        ("html_script", "<script>alert('hack')</script>", "HTML脚本注入"),
        ("sql_inject", "admin' or 1=1--", "SQL注入字符"),
        # 4. 乱码 & 异常编码
        ("garbage_code", "锟斤拷烫烫烫", "中文乱码"),
        ("mix_lang", "こんにちは 你好 123 !!!", "多语言混合"),
        # 5. 极端短/边界
        ("single_char", "a", "单个字符"),
        ("only_number", "123456", "纯数字文本"),
        ("only_emoji", "😀😃😄😁", "纯emoji表情"),
    ]
    return cases

# ====================== 测试类（Allure 模块分级） ======================
@allure.feature("AI安全与合规")
@allure.story("脏输入鲁棒性 + 模型公平性审计")
class TestAISecurityCompliance:
    """
    AI上线合规底线测试：
    1. 脏输入鲁棒性：异常输入不崩溃、不越权、输出合法
    2. 公平性审计：不同群体性能差异可控，满足合规要求
    """

    # ====================== 第一部分：脏输入鲁棒性测试 ======================
    @allure.severity("blocker")
    @allure.title("脏输入鲁棒性校验：各类异常输入模型不崩溃")
    @pytest.mark.parametrize("case_name,input_text,case_desc", get_dirty_input_cases())
    def test_dirty_input_robust(self, case_name, input_text, case_desc):
        """
        用例说明：
        校验模型在空值、超长、特殊字符、注入、乱码等脏输入下：
        1. 不抛出异常、不崩溃
        2. 输出格式合法（仅0/1）
        """
        with allure.step(f"步骤1：输入脏数据 [{case_desc}]"):
            test_input = [input_text] if input_text is not None else [""]

        with allure.step("步骤2：执行模型预测"):
            try:
                pred = model.predict(test_input)
                success_flag = True
            except Exception as e:
                success_flag = False
                # 异常日志写入附件
                allure.attach(
                    str(e),
                    name=f"异常日志_{case_name}",
                    attachment_type=allure.attachment_type.TEXT
                )

        with allure.step("步骤3：断言鲁棒性（核心门禁）"):
            # 断言1：模型不能崩溃
            assert success_flag is True, f"脏输入 [{case_desc}] 导致模型异常崩溃"
            # 断言2：输出必须为合法分类（0/1）
            assert pred[0] in [0, 1], f"脏输入 [{case_desc}] 输出非法结果: {pred[0]}"

    @allure.severity("critical")
    @allure.title("模糊测试：随机海量脏输入稳定性压测")
    def test_fuzz_stability(self):
        """随机生成大量脏字符串,压测模型长期稳定性"""
        fuzz_times = 20
        error_count = 0
        error_logs = []

        with allure.step(f"循环执行 {fuzz_times} 次模糊测试"):
            for i in range(fuzz_times):
                fuzz_str = generate_random_fuzz_str(random.randint(10, 500))
                try:
                    model.predict([fuzz_str])
                except Exception as e:
                    error_count += 1
                    error_logs.append(f"第{i+1}轮失败: {str(e)}")

        with allure.step("步骤2：校验模糊测试错误率"):
            if error_logs:
                allure.attach(
                    "\n".join(error_logs),
                    name="模糊测试错误汇总",
                    attachment_type=allure.attachment_type.TEXT
                )
            # 门禁：模糊测试错误数必须为0
            assert error_count == 0, f"模糊测试出现 {error_count} 次异常，模型鲁棒性不达标"

    # ====================== 第二部分：模型公平性审计（合规核心） ======================
    @allure.severity("critical")
    @allure.title("公平性审计：不同群体准确率差异校验")
    def test_fairness_accuracy_balance(self):
        """按敏感特征分组,校验各组准确率差异在合规阈值内"""
        X_text = fairness_data["text"].tolist()
        y_true = fairness_data["label"].tolist()
        y_pred = model.predict(X_text)
        sensitive_age = fairness_data["age_group"]
        sensitive_gender = fairness_data["gender"]

        with allure.step("计算【年龄分组】各群体准确率"):
            mf_age = MetricFrame(
                metrics=accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_age
            )
            age_acc = mf_age.by_group.to_dict()
            age_overall = mf_age.overall

        with allure.step("计算【性别分组】各群体准确率"):
            mf_gender = MetricFrame(
                metrics=accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_gender
            )
            gender_acc = mf_gender.by_group.to_dict()
            gender_overall = mf_gender.overall

        # 写入报告附件
        report_content = f"""
整体准确率: {age_overall:.4f}
年龄分组准确率: {age_acc}
性别分组准确率: {gender_acc}
"""
        allure.attach(
            report_content,
            name="群体准确率报表",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("断言：各组准确率差异 ≤ 5%（合规阈值）"):
            # 年龄组差异校验
            age_diff = abs(age_acc["young"] - age_acc["old"])
            assert age_diff <= 0.05, f"年龄群体准确率差异 {age_diff:.2%} > 5%，存在偏见风险"
            # 性别组差异校验
            gender_diff = abs(gender_acc["male"] - gender_acc["female"])
            assert gender_diff <= 0.05, f"性别群体准确率差异 {gender_diff:.2%} > 5%，存在偏见风险"

    @allure.severity("critical")
    @allure.title("公平性审计：人口均等 & 等机会指标校验")
    def test_fairness_legal_indicators(self):
        """计算行业合规指标：人口 parity、等机会差异，阻断高偏见模型上线"""
        X_text = fairness_data["text"].tolist()
        y_true = fairness_data["label"].tolist()
        y_pred = model.predict(X_text)
        sensitive_gender = fairness_data["gender"]

        with allure.step("计算人口均等差异 (Demographic Parity)"):
            dp_diff = demographic_parity_difference(
                y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_gender
            )

        with allure.step("计算等机会差异 (Equalized Odds)"):
            eo_diff = equalized_odds_difference(
                y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_gender
            )

        # 指标附件
        indicator_text = f"""
人口均等差异 DP: {dp_diff:.4f}
等机会差异 EO: {eo_diff:.4f}
合规阈值: 两项均 ≤ 0.1
"""
        allure.attach(
            indicator_text,
            name="公平性合规指标",
            attachment_type=allure.attachment_type.TEXT
        )

        with allure.step("合规门禁断言"):
            assert abs(dp_diff) <= 0.1, f"人口均等差异 {dp_diff:.2%} 超标，不合规"
            assert abs(eo_diff) <= 0.1, f"等机会差异 {eo_diff:.2%} 超标，不合规"