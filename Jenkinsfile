pipeline {
    agent any
    environment {
        PROJECT_DIR = "${WORKSPACE}"
        ALLURE_RESULTS = "${WORKSPACE}/allure-results"
        PYTHONPATH = "/home/ubuntu/.local/lib/python3.10/site-packages"
    }
    stages {
        stage('0. 安装依赖') {
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_DIR}
                    /usr/bin/python3 -m pip install --no-cache-dir -r requirements.txt --user
                    /usr/bin/python3 -m pip uninstall -y evidently
                    /usr/bin/python3 -m pip install -U evidently>=0.14.0 --user --no-cache-dir
                    
                    # 关键：刷新包缓存
                    /usr/bin/python3 -c "import site; site.main()"
                '''
            }
        }

        stage('1. 检查环境') {
            steps {
                sh '''
                    whoami
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