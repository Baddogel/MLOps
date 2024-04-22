pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Create Data') {
            steps {
                sh 'python3 data_creation.py'
            }
        }
        stage('Preprocessing') {
            steps {
                sh 'python3 model_preprocessing.py'
            }
        }
        stage('Model learning') {
            steps {
                sh 'python3 model_preparation.py'
            }
        }
        stage('Model testing') {
            steps {
                sh 'python3 model_testing.py'
            }
        }
    }
    post {
        always {
            echo 'ML pipeline completed.'
        }
    }
}