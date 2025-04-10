from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def init_pinecone():
    """Initialize Pinecone with new syntax"""
    # Create Pinecone instance
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Print first 5 chars of API key for verification (never print full key)
    current_key = os.getenv("PINECONE_API_KEY")
    print(f"Using API key starting with: {current_key[:5]}...")
    
    # List current indexes
    current_indexes = pc.list_indexes()
    print(f"Current indexes in account: {current_indexes.names()}")
    
    return True

def create_test_index():
    """Create a test index using new syntax"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = 'test-index'
    
    # Check if index already exists
    if index_name not in pc.list_indexes().names():
        # Create new index
        pc.create_index(
            name=index_name,
            dimension=1536,  # adjust dimension as needed
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # adjust region as needed
            )
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Index {index_name} already exists")
    
    return True

def cleanup_test_index():
    """Clean up test index"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = 'test-index'
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        print(f"Deleted index: {index_name}")
    
    return True

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('pinecone_test_dag',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:
    
    init_task = PythonOperator(
        task_id='init_pinecone',
        python_callable=init_pinecone
    )
    
    create_index_task = PythonOperator(
        task_id='create_test_index',
        python_callable=create_test_index
    )
    
    cleanup_task = PythonOperator(
        task_id='cleanup_test_index',
        python_callable=cleanup_test_index
    )
    
    # Set task dependencies
    init_task >> create_index_task >> cleanup_task