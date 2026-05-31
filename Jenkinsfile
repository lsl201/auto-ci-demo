pipeline {
    agent any
    environment {
        ALLURE_RESULTS = "allure-results"
    }
    stages {
        stage('0. 安装依赖') {
            steps {
                sh '''
                    set -e
                    /usr/bin/python3 -m pip install --upgrade pip
                    /usr/bin/python3 -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh '/usr/bin/python3 -m pip show evidently'
                sh '/usr/bin/python3 -m pip show numpy'
            }
        }

        stage('2. 运行测试（双保险：目录隔离 + marker）') {
            steps {
                sh '''
                    # 1. 清空上次结果，避免残留
                    rm -rf ${ALLURE_RESULTS}
                    
                    # 2. 只扫描 tests/debug/ 目录 + 只跑 @mine 标记的用例
                    /usr/bin/python3 -m pytest tests/debug/ -m "mine" -v --alluredir=${ALLURE_RESULTS}
                '''
            }
        }
    }
    post {
        always {
            allure results: [[path: "${ALLURE_RESULTS}"]]
        }
    }
}