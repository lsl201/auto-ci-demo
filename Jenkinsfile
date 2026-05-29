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
                    
                    # 强制彻底删除旧版
                    /usr/bin/python3 -m pip uninstall -y evidently
                    
                    # 强制安装新版！！！关键在这里！！
                    /usr/bin/python3 -m pip install -U evidently==0.7.21 --user --no-cache-dir
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh '''
                    /usr/bin/python3 -m pip show evidently
                '''
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