pipeline {
    agent any
    environment {
        PROJECT_DIR = "/home/ubuntu/Desktop/auto-ci-demo"
        MODEL_DIR   = "/home/ubuntu/Desktop/ai-models"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        stage('0. 安装依赖（强制成功）') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}
                    # 不用 sudo，避免卡住；安装到系统 Python3
                    python3 -m pip install --upgrade pip
                    python3 -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('1. 验证 Evidently 安装') {
            steps {
                sh '''
                    set -e
                    python3 -m pip show evidently
                    python3 -c "from evidently.report import Report; print('✅ Evidently 导入成功')"
                '''
            }
        }

        stage('2. 环境与目录检查') {
            steps {
                sh '''
                    whoami
                    echo "PROJECT_DIR=${PROJECT_DIR}"
                    ls -la ${PROJECT_DIR}
                '''
            }
        }

        stage('3. 执行全量自动化测试') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}
                    python3 -m pytest \
                        test_demo_case.py test_model_base.py test_model_offline.py \
                        -v --alluredir=${ALLURE_RESULTS} --clean-alluredir
                '''
            }
        }
    }
    post {
        always {
            allure([
                includeProperties: false,
                jdk: '',
                properties: [],
                reportBuildPolicy: 'ALWAYS',
                results: [[path: "${ALLURE_RESULTS}"]]
            ])
        }
    }
}