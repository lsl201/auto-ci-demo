pipeline {
    agent any
    environment {
        PROJECT_DIR = "${WORKSPACE}"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        stage('0. 修复环境 & 安装依赖') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}
                    
                    # 🔥 关键：强制固定 pip 版本，避开 Ubuntu 错误
                    python3 -m pip install pip==24.0 --user --ignore-installed
                    
                    # 正常安装依赖
                    python3 -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh 'whoami'
                sh 'ls -la ${PROJECT_DIR}'
            }
        }

        stage('2. 运行测试') {
            steps {
                sh '''
                    cd ${PROJECT_DIR}
                    python3 -m pytest -v --alluredir=${ALLURE_RESULTS}
                '''
            }
        }
    }
    post {
        always {
            allure(results: [[path: "${ALLURE_RESULTS}"]])
        }
    }
}