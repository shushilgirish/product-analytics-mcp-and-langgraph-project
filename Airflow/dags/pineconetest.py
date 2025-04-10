from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pinecone
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

def verify_pinecone_connection():
    """Verify Pinecone connection and configuration"""
    # Print first 5 chars of API key for verification (never print full key)
    current_key = os.getenv("PINECONE_API_KEY")
    print(f"Using API key starting with: {current_key[:5]}...")
    
    # Initialize Pinecone
    pinecone.init(api_key=current_key, environment='us-west1-gcp')  # adjust environment as needed
    
    # List current indexes
    current_indexes = pinecone.list_indexes()
    print(f"Current indexes in account: {current_indexes}")
    
    return True

def cleanup_old_connection():
    """Clean up any existing Pinecone connections"""
    try:
        pinecone.deinit()  # This will clear any existing connections
    except:
        pass
    return True

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Initialize Pinecone
def init_pinecone():
    pinecone.init(api_key=api_key, environment='us-west1-gcp')  # Adjust the environment as needed

# Create an index
def create_index():
    index_name = 'test-index'
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=128)  # Adjust dimension as needed

# Insert data into the index
def insert_data():
    index = pinecone.Index('test-index')
    vectors = [
        ("id1", [0.1, 0.2, 0.3, ...]),  # Replace with actual vector data
        ("id2", [0.4, 0.5, 0.6, ...]),
        # Add more vectors as needed
    ]
    index.upsert(vectors)

# Query the index
def query_index():
    index = pinecone.Index('test-index')
    query_result = index.query([0.1, 0.2, 0.3, ...], top_k=3)  # Replace with actual query vector
    print(query_result)

# Clean up the index
def cleanup_index():
    pinecone.delete_index('test-index')

# Define the DAG
with DAG('pinecone_test_dag', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    init_task = PythonOperator(
        task_id='init_pinecone',
        python_callable=init_pinecone
    )

    create_index_task = PythonOperator(
        task_id='create_index',
        python_callable=create_index
    )

    insert_data_task = PythonOperator(
        task_id='insert_data',
        python_callable=insert_data
    )

    query_index_task = PythonOperator(
        task_id='query_index',
        python_callable=query_index
    )

    cleanup_index_task = PythonOperator(
        task_id='cleanup_index',
        python_callable=cleanup_index
    )

    # Set task dependencies
    init_task >> create_index_task >> insert_data_task >> query_index_task >> cleanup_index_task