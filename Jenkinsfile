pipeline {
    agent any
    environment {
        PROJECT_DIR = "${WORKSPACE}"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
    }
    stages {
        stage('0. 清理坏缓存 + 安装依赖') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}

                    # 🔥 强制清理坏掉的本地pip
                    rm -rf ~/.local/lib/python3.10/site-packages/pip*

                    # 重装正确版本
                    python3 -m ensurepip --upgrade --user
                    python3 -m pip install pip==24.0 --user

                    # 安装依赖
                    python3 -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh 'whoami && ls -la'
            }
        }

        stage('2. 运行测试') {
            steps {
                sh '''
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