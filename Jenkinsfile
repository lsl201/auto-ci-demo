pipeline {
    agent any
    environment {
        PROJECT_DIR = "/home/ubuntu/Desktop/auto-ci-demo"
        MODEL_DIR   = "/home/ubuntu/Desktop/ai-models"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        // 0. 安装依赖（锁死版本 + 强制系统环境）
        stage('0. 安装依赖（锁死版本）') {
            steps {
                sh '''
                    cd ${PROJECT_DIR}
                    sudo python3 -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        // 检查 Evidently 版本
        stage('检查 Evidently 版本') {
            steps {
                sh 'python3 -m pip show evidently'
            }
        }

        stage('1. 环境与目录检查') {
            steps {
                sh 'whoami'
                sh 'echo 项目目录：${PROJECT_DIR}'
                sh 'echo 模型目录：${MODEL_DIR}'
                sh 'ls -la ${PROJECT_DIR}'
            }
        }

        stage('2. 执行全量自动化测试') {
            steps {
                sh '''
                    cd ${PROJECT_DIR}
                    python3 -m pytest -v --alluredir=${ALLURE_RESULTS} --clean-alluredir
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