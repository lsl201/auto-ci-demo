pipeline {
    agent any
    environment {
        PROJECT_DIR = "/home/ubuntu/Desktop/auto-ci-demo"
        MODEL_DIR   = "/home/ubuntu/Desktop/ai-models"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
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