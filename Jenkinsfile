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

                    # ===================== 【关键修改】 =====================
                    # 卸载旧版，安装你要求的：0.5.0（不是0.7.21）
                    # =========================================================
                    /usr/bin/python3 -m pip uninstall -y evidently
                    /usr/bin/python3 -m pip install -U evidently==0.5.0 --user --no-cache-dir
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh '/usr/bin/python3 -m pip show evidently'
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