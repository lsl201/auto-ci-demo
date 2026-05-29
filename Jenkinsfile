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

        stage('2. 运行测试') {
            steps {
                sh '''
            /usr/bin/python3 -m pytest \
              --testpaths my_cases \
              -v --alluredir=allure-results
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