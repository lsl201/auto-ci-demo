pipeline {
    agent any
    environment {
        PROJECT_DIR = "/home/ubuntu/Desktop/auto-ci-demo"
        MODEL_DIR   = "/home/ubuntu/Desktop/ai-models"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        // 新增：0. 安装依赖（锁死版本）
        stage('0. 安装依赖（锁死版本）') {
            steps {
                sh '''
                    cd ${PROJECT_DIR}
                    # 严格按 requirements.txt 安装，不升级、不兼容
                    pip install --no-cache-dir -r requirements.txt
                '''
            }
        }
        // 新增：检查 Evidently 版本
        stage('检查 Evidently 版本') {
            steps {
                sh 'pip show evidently'
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
                    pytest -v --alluredir=${ALLURE_RESULTS} --clean-alluredir
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