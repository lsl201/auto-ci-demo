# tests/metrics/test_99_final_attach_csv.py
import os
import subprocess
import allure
import pytest

# 项目根目录（写死，避免相对路径乱跳）
ROOT = "/home/ubuntu/Desktop/auto-ci-demo"
SCRIPT_PATH = os.path.join(ROOT, "scripts/metric_summary.py")
CSV_PATH = os.path.join(ROOT, "metrics_summary.csv")

def run_summary_and_attach():
    # 1. 先打印路径，方便排查
    print(f"🔍 脚本路径: {SCRIPT_PATH}")
    print(f"🔍 CSV 路径: {CSV_PATH}")
    print(f"🔍 脚本存在？{os.path.exists(SCRIPT_PATH)}")

    # 2. 执行汇总脚本（不存在也不抛错，只警告）
    if os.path.exists(SCRIPT_PATH):
        try:
            result = subprocess.run(
                ["python3", SCRIPT_PATH],
                check=True,
                stdout=subprocess.PIPE,  # 只捕获输出，不拦截
                text=True
            )
            print(result.stdout)
            print("✅ 汇总脚本执行成功")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"❌ 汇总脚本执行失败：{e.stderr}")
    else:
        pytest.fail(f"❌ 汇总脚本不存在：{SCRIPT_PATH}")

    # 3. 检查并附加 CSV
    if os.path.exists(CSV_PATH):
        allure.attach.file(
            source=CSV_PATH,
            name="MLflow全版本指标汇总表",
            attachment_type=allure.attachment_type.CSV
        )
        print("📎 CSV 已嵌入 Allure 报告")
    else:
        pytest.fail(f"❌ CSV 文件不存在：{CSV_PATH}")

# 最后执行：用 order=-1，不要用 @pytest.mark.last
@pytest.mark.order(-1)
@pytest.mark.continue_on_fail   # 👈 加这个
def test_final_summary_attach():
    with allure.step("执行 MLflow 指标汇总并附加 CSV"):
        run_summary_and_attach()