pipeline {
    agent any
    environment {
        PROJECT_DIR = "${WORKSPACE}"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        stage('0. 安装依赖（纯净稳定版）') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}

                    # 直接用系统自带的稳定 pip，不升级、不破坏系统
                    python3 -m pip install --no-cache-dir -r requirements.txt --user
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh 'whoami && ls -la ${PROJECT_DIR}'
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