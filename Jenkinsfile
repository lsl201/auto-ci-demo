pipeline {
    agent any
    environment {
        PROJECT_DIR = "/home/ubuntu/Desktop/auto-ci-demo"
        ALLURE_RESULTS = "${PROJECT_DIR}/allure-results"
        MLFLOW_TRACKING_URI = "file:///${PROJECT_DIR}/mlruns"
    }
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(daysToKeepStr: '30'))
    }
    stages {
        stage('0. 环境校验') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                        set -e
                        echo "=== 工作目录: $(pwd)"
                        echo "=== MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
                        mkdir -p mlruns
                        mkdir -p ${ALLURE_RESULTS}
                        chmod -R 777 /home/ubuntu/Desktop/auto-ci-demo
                        echo "✅ 目录准备完成"
                    '''
                }
            }
        }

        stage('1. 安装依赖') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                        set -e
                        /usr/bin/python3 -m pip install --upgrade pip
                        /usr/bin/python3 -m pip install --no-cache-dir -r requirements.txt
                    '''
                }
            }
        }

        stage('2. 执行所有测试（metrics + robustness）') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                        set +e
                        # 导出全部环境变量
                        export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

                        # 执行pytest
                        pytest tests/metrics/ tests/robustness/ \
                            -v -s --tb=short \
                            --alluredir=${ALLURE_RESULTS} \
                            --clean-alluredir

                        # 捕获pytest执行返回码
                        PYTEST_EXIT_CODE=$?
                        set -e

                        # 用例失败：直接退出，不再执行CSV汇总、模型注册
                        if [ $PYTEST_EXIT_CODE -ne 0 ];then
                            echo "❌ 测试用例执行失败，跳过指标校验和模型注册"
                            exit $PYTEST_EXIT_CODE
                        fi

                        # 用例全部成功，才运行汇总脚本（内嵌执行，不再单独stage）
                        echo "📊 开始生成最终版 metrics_summary.csv ..."
                        python3 scripts/metric_summary.py
                        echo "✅ CSV已最新，与MLflow完全对齐"
                    '''
                }
            }
        }

        stage('3. 验证 MLflow 数据') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh '''
                        echo "===== MLflow 目录内容 ====="
                        ls -la mlruns/
                        ls -la mlruns/*/
                    '''
                }
            }
        }
    }
}
    }
    post {
        always {
            allure includeProperties: false, jdk: '', results: [[path: "${ALLURE_RESULTS}"]]
        }
        success {
            echo "✅ 构建成功！MLflow 数据已写入桌面"
        }
        failure {
            echo "❌ 构建失败"
        }
    }
}