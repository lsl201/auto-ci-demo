# AI模型Pytest测试目录说明
## 目录分工
1. smoke：冒烟测试
    作用：CI流水线首轮快速校验，少量样本跑通预测链路，失败直接阻断全量测试
    执行：pytest tests/smoke/ -v
2. metrics：常规指标测试
    作用：准确率/FPR/FNR离线效果验收，指标入库MLflow
    执行：pytest tests/metrics/ -v
3. drift：数据漂移测试
    作用：Evidently对比训练集&线上数据分布，漂移分数写入MLflow
    执行：pytest tests/drift/ -v
4. regression：全量回归测试
    作用：版本迭代全量复测（常规+漂移+鲁棒合集），指标下滑超标拦截上线
    执行：pytest tests/regression/ -v
5. robustness【新增】：鲁棒性专项
    子用例：脏输入、OOD域外、对抗注入扰动测试
    判定：鲁棒指标低于基线90% → 测试不通过
    执行：pytest tests/robustness/ -v -m robust_noise

## 全量执行
pytest tests/ -v --alluredir=allure-results