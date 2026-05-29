pipeline {
    agent any
    environment {
        PROJECT_DIR = "${WORKSPACE}"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        stage('0. 安装依赖') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}
                    /usr/bin/python3 -m pip install --no-cache-dir -r requirements.txt --user
                    /usr/bin/python3 -m pip install -U evidently>=0.14.0 --user
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
                    /usr/bin/python3 -m pytest -v --alluredir=${ALLURE_RESULTS}
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