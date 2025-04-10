"""
Marketing Book Processing Pipeline: PDF → Markdown → Vectors → Pinecone
Features comprehensive debugging, analytics, and error recovery mechanisms.

Book: S3D7W4_Marketing_Management.pdf (833+ pages)
Vector dimensions: 384
"""
import json
from utils.mistralparsing_userpdf import process_pdf
from utils.chunking import KamradtModifiedChunker
import pinecone
import os
import json
import time
import uuid
import tempfile
import shutil
import traceback
import re
import logging
import psutil
import numpy as np
import requests
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowFailException
from airflow.models import Variable, TaskInstance

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'book.json')
with open(config_path, 'r') as f:
    config = json.load(f)
AWS_CONN_ID = config["AWS_CONN_ID"]
S3_BUCKET = config["S3_BUCKET"]  
S3_BASE_FOLDER = config["S3_BASE_FOLDER"]

# =====================================================================================
# CONFIGURATION SECTION
# =====================================================================================

# Book information
BOOK_NAME = config["BOOK_NAME"]  # Book name for tracking
BOOK_S3_PATH = config["BOOK_S3_PATH"]  # S3 path to the book
BOOK_LOCAL_PATH = config["BOOK_LOCAL_PATH"]  # Local path to the book (if available)

S3_MARKDOWN_FOLDER = f"{S3_BASE_FOLDER}/markdown"
S3_VECTOR_FOLDER = f"{S3_BASE_FOLDER}/vectors"
S3_DIAGNOSTIC_FOLDER = f"{S3_BASE_FOLDER}/diagnostics"

# Processing configuration
BATCH_SIZE = 50  # Pages per batch
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4  # For parallel processing
EMBEDDING_DIMENSION = 1536  # ✅ CHANGE: Updated for OpenAI embeddings
PINECONE_INDEX_NAME =  config["PINECONE_INDEX_NAME"]  # ✅ CHANGE: Added index name
PINECONE_NAMESPACE = "book-kotler"  # ✅ CHANGE: Added namespace
# Default arguments for DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 9),
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

# =====================================================================================
# MONITORING FRAMEWORK
# =====================================================================================

class ProcessingMetrics:
    """
    Comprehensive monitoring framework for pipeline telemetry.
    Tracks performance, errors, and resource usage throughout the pipeline.
    """
    def __init__(self, book_name: str):
        self.book_name = book_name
        self.start_time = time.time()
        self.stages = {}
        self.page_metrics = {}
        self.chunk_metrics = {}
        self.errors = []
        self.warnings = []
        self.memory_samples = []
        self.sample_memory()  # Initial memory sample
        
    def sample_memory(self) -> float:
        """Capture current memory usage and add to tracking history"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.memory_samples.append((time.time(), memory_mb))
        return memory_mb
        
    def start_stage(self, stage_name: str, metadata: Dict = None) -> None:
        """Begin timing a pipeline stage with optional context"""
        self.stages[stage_name] = {
            'start_time': time.time(),
            'status': 'running',
            'memory_before': self.sample_memory(),
            'metadata': metadata or {}
        }
        logging.info(f"📊 Starting stage: {stage_name}")
        
    def end_stage(self, stage_name: str, status: str = 'success', metrics: Dict = None) -> Dict:
        """End timing for a stage and record performance metrics"""
        if stage_name not in self.stages:
            self.warning(f"Attempted to end unknown stage: {stage_name}")
            return {}
            
        end_time = time.time()
        memory_after = self.sample_memory()
        duration = end_time - self.stages[stage_name]['start_time']
        memory_change = memory_after - self.stages[stage_name]['memory_before']
        
        self.stages[stage_name].update({
            'end_time': end_time,
            'duration': duration,
            'memory_after': memory_after,
            'memory_change': memory_change,
            'status': status,
            'metrics': metrics or {}
        })
        
        # Log completion with key metrics
        log_method = logging.info if status == 'success' else logging.warning
        log_method(f"📊 Completed stage: {stage_name} ({status}) in {duration:.2f}s with {memory_change:.2f}MB memory change")
        
        return self.stages[stage_name]
    
    def record_page_metrics(self, page_num: int, extraction_time: float, 
                           char_count: int, validation_score: float) -> None:
        """Record detailed metrics for individual page processing"""
        self.page_metrics[page_num] = {
            'extraction_time': extraction_time,
            'char_count': char_count,
            'validation_score': validation_score,
            'extraction_rate': char_count / extraction_time if extraction_time > 0 else 0
        }
    
    def record_chunk_metrics(self, chunk_id: str, text_length: int, embedding_time: float,
                            token_count: int = None, embedding_norm: float = None) -> None:
        """Record metrics for text chunking and embedding"""
        self.chunk_metrics[chunk_id] = {
            'text_length': text_length,
            'token_count': token_count,
            'embedding_time': embedding_time,
            'embedding_norm': embedding_norm,
            'embedding_rate': text_length / embedding_time if embedding_time > 0 else 0
        }
    
    def error(self, stage: str, error_msg: str, error_obj: Exception = None,
             context: Dict = None) -> None:
        """Record structured error information with context"""
        error_entry = {
            'timestamp': time.time(),
            'stage': stage,
            'error': error_msg,
            'error_type': type(error_obj).__name__ if error_obj else 'Unknown',
            'context': context or {},
        }
        
        # Add traceback if exception provided
        if error_obj:
            error_entry['traceback'] = traceback.format_exception(
                type(error_obj), error_obj, error_obj.__traceback__
            )
        
        self.errors.append(error_entry)
        logging.error(f"❌ ERROR in {stage}: {error_msg}")
    
    def warning(self, message: str, context: Dict = None) -> None:
        """Record non-critical issues and anomalies"""
        self.warnings.append({
            'timestamp': time.time(),
            'message': message,
            'context': context or {}
        })
        logging.warning(f"⚠️ WARNING: {message}")
    
    def get_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        duration = time.time() - self.start_time
        
        # Calculate aggregate metrics
        processed_pages = len(self.page_metrics)
        processed_chunks = len(self.chunk_metrics)
        
        # Calculate avg/min/max for pages and chunks if data exists
        page_stats = {}
        if self.page_metrics:
            extraction_times = [m['extraction_time'] for m in self.page_metrics.values()]
            char_counts = [m['char_count'] for m in self.page_metrics.values()]
            
            page_stats = {
                'avg_extraction_time': sum(extraction_times) / len(extraction_times),
                'min_extraction_time': min(extraction_times),
                'max_extraction_time': max(extraction_times),
                'avg_char_count': sum(char_counts) / len(char_counts),
                'total_char_count': sum(char_counts)
            }
        
        chunk_stats = {}
        if self.chunk_metrics:
            embedding_times = [m['embedding_time'] for m in self.chunk_metrics.values() if 'embedding_time' in m]
            text_lengths = [m['text_length'] for m in self.chunk_metrics.values()]
            
            if embedding_times:
                chunk_stats = {
                    'avg_embedding_time': sum(embedding_times) / len(embedding_times),
                    'total_embedding_time': sum(embedding_times),
                    'avg_chunk_length': sum(text_lengths) / len(text_lengths),
                    'min_chunk_length': min(text_lengths),
                    'max_chunk_length': max(text_lengths)
                }
        
        # Get memory stats
        if self.memory_samples:
            memory_values = [m for _, m in self.memory_samples]
            memory_stats = {
                'peak_memory_mb': max(memory_values),
                'initial_memory_mb': self.memory_samples[0][1],
                'final_memory_mb': self.memory_samples[-1][1],
                'memory_change_mb': self.memory_samples[-1][1] - self.memory_samples[0][1]
            }
        else:
            memory_stats = {}
        
        # Build summary dict
        return {
            'book_name': self.book_name,
            'duration_seconds': duration,
            'pages_processed': processed_pages,
            'chunks_created': processed_chunks,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'completed_stages': [s for s, details in self.stages.items() if details.get('status') == 'success'],
            'failed_stages': [s for s, details in self.stages.items() if details.get('status') == 'failed'],
            'page_stats': page_stats,
            'chunk_stats': chunk_stats,
            'memory_stats': memory_stats,
            'processing_rate_pages_per_minute': (processed_pages / duration) * 60 if duration > 0 else 0
        }
    
    def generate_report(self, include_details: bool = True) -> Dict:
        """Generate comprehensive report with optional detail level"""
        summary = self.get_summary()
        
        # Create full report with all details
        if include_details:
            report = {
                'summary': summary,
                'stages': self.stages,
                'errors': self.errors,
                'warnings': self.warnings
            }
            
            # Include page and chunk metrics if not too large
            if len(self.page_metrics) <= 100:
                report['page_metrics'] = self.page_metrics
            else:
                report['page_metrics'] = {
                    'sample': {k: self.page_metrics[k] for k in list(self.page_metrics.keys())[:100]},
                    'count': len(self.page_metrics)
                }
                
            if len(self.chunk_metrics) <= 100:
                report['chunk_metrics'] = self.chunk_metrics
            else:
                report['chunk_metrics'] = {
                    'sample': {k: self.chunk_metrics[k] for k in list(self.chunk_metrics.keys())[:100]},
                    'count': len(self.chunk_metrics)
                }
        else:
            # Create summary-only report
            report = summary
            
        return report

    def save_report(self, s3_hook: S3Hook, report_name: str) -> str:
        """Save diagnostic report to S3 for analysis"""
        report = self.generate_report()
        
        # Create temp file for the report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(report, temp_file, indent=2, default=str)
            temp_path = temp_file.name
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{S3_DIAGNOSTIC_FOLDER}/{report_name}_{timestamp}.json"
        
        s3_hook.load_file(
            filename=temp_path,
            key=s3_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        logging.info(f"📊 Diagnostic report saved to s3://{S3_BUCKET}/{s3_key}")
        return s3_key

# Initialize global metrics tracker
metrics = ProcessingMetrics(BOOK_NAME)

# =====================================================================================
# QUALITY VALIDATION FRAMEWORK
# =====================================================================================

class QualityValidator:
    """
    Validates quality at different stages of the pipeline:
    1. PDF extraction quality
    2. Markdown conversion quality
    3. Chunking quality
    4. Embedding quality
    """
    
    @staticmethod
    def validate_extracted_text(page_text: str, page_num: int) -> Tuple[float, List[str]]:
        """
        Validate quality of extracted text from PDF page
        Returns (quality_score, issues)
        """
        issues = []
        
        # Check for empty content
        if not page_text or len(page_text.strip()) == 0:
            issues.append(f"Page {page_num} has no text content")
            return 0.0, issues
            
        # Check for minimal content (likely extraction failure)
        if len(page_text.strip()) < 100:
            issues.append(f"Page {page_num} has suspiciously short content ({len(page_text)} chars)")
            
        # Check for garbage characters (encoding issues)
        garbage_char_count = len(re.findall(r'[^\x00-\x7F\u00A0-\u00FF\u0100-\u017F\u0180-\u024F]', page_text))
        garbage_ratio = garbage_char_count / max(1, len(page_text))
        if garbage_ratio > 0.05:  # More than 5% garbage characters
            issues.append(f"Page {page_num} has high ratio of garbage characters ({garbage_ratio:.2%})")
            
        # Check for common PDF extraction artifacts
        if re.search(r'\f', page_text):
            issues.append(f"Page {page_num} contains form feed characters")
            
        # Check for missing spaces between words (common PDF extraction issue)
        long_word_count = len(re.findall(r'\b\w{25,}\b', page_text))
        if long_word_count > 0:
            issues.append(f"Page {page_num} has {long_word_count} abnormally long words (possible extraction error)")
            
        # Calculate overall quality score
        if not issues:
            quality_score = 1.0  # Perfect
        else:
            # Deduct points for each type of issue
            quality_score = max(0.0, 1.0 - (len(issues) * 0.2))
            
        return quality_score, issues
    
    @staticmethod
    def validate_markdown_conversion(original_text: str, markdown_text: str) -> Tuple[float, Dict]:
        """
        Validate quality of markdown conversion
        Returns (quality_score, metrics)
        """
        metrics = {}
        
        # Length comparison - should be reasonably similar
        orig_len = len(original_text.strip())
        md_len = len(markdown_text.strip())
        len_ratio = md_len / orig_len if orig_len > 0 else 0
        metrics['length_ratio'] = len_ratio
        
        # Check for expected markdown structures (headers, lists, etc.)
        metrics['has_headers'] = len(re.findall(r'^#{1,6}\s+', markdown_text, re.MULTILINE)) > 0
        metrics['has_lists'] = len(re.findall(r'^\s*[*-]\s+', markdown_text, re.MULTILINE)) > 0
        metrics['has_tables'] = '|' in markdown_text and len(re.findall(r'^\|.*\|$', markdown_text, re.MULTILINE)) > 0
        
        # Calculate similarity using character-level comparison
        # This is a simple approach - more sophisticated NLP methods could be used
        char_overlap = sum(1 for c in markdown_text if c in original_text) 
        char_similarity = char_overlap / max(1, len(markdown_text))
        metrics['char_similarity'] = char_similarity
        
        # Calculate overall quality score (weighted factors)
        score = 0.0
        # Length ratio should be close to 1.0
        if 0.8 <= len_ratio <= 1.2:
            score += 0.5
        elif 0.5 <= len_ratio <= 1.5:
            score += 0.3
        
        # Character similarity
        score += min(0.5, char_similarity)
        
        # Adjust for markup features
        if metrics['has_headers']:
            score = min(1.0, score + 0.1)
            
        return score, metrics
    
    @staticmethod
    def validate_text_chunks(original_text: str, chunks: List[str]) -> Tuple[float, Dict]:
        """
        Validate quality of text chunking
        Returns (quality_score, metrics)
        """
        metrics = {}
        
        # Check content coverage (all original content should be in chunks)
        total_chunk_length = sum(len(c) for c in chunks)
        original_length = len(original_text)
        
        # Account for overlap in calculation
        estimated_overlap = (len(chunks) - 1) * CHUNK_OVERLAP if len(chunks) > 1 else 0
        adjusted_chunk_length = max(1, total_chunk_length - estimated_overlap)
        
        coverage_ratio = adjusted_chunk_length / original_length if original_length > 0 else 0
        metrics['coverage_ratio'] = coverage_ratio
        
        # Analyze chunk size distribution
        chunk_lengths = [len(c) for c in chunks]
        metrics['chunk_count'] = len(chunks)
        if chunk_lengths:
            metrics['min_chunk_size'] = min(chunk_lengths)
            metrics['max_chunk_size'] = max(chunk_lengths) 
            metrics['avg_chunk_size'] = sum(chunk_lengths) / len(chunk_lengths)
            metrics['chunk_size_stddev'] = np.std(chunk_lengths) if len(chunk_lengths) > 1 else 0
        
        # Check boundary quality (prefer breaking at paragraphs/sentences)
        good_boundaries = 0
        for chunk in chunks:
            # Check if chunk starts with paragraph, list item, or heading
            if re.match(r'^(#+\s+|\s*[*-]\s+|\w)', chunk):
                good_boundaries += 1
                
            # Check if chunk ends with sentence-ending punctuation
            if re.search(r'[.!?]\s*$', chunk):
                good_boundaries += 1
                
        boundary_quality = good_boundaries / (len(chunks) * 2) if chunks else 0
        metrics['boundary_quality'] = boundary_quality
        
        # Calculate overall quality score
        score = 0.0
        
        # Coverage is critical
        if coverage_ratio >= 0.98:
            score += 0.6
        elif coverage_ratio >= 0.9:
            score += 0.4
        elif coverage_ratio >= 0.8:
            score += 0.2
            
        # Boundary quality
        score += boundary_quality * 0.4
            
        return score, metrics
    
    @staticmethod
    def validate_embeddings(chunk_texts: List[str], embeddings: List[List[float]]) -> Tuple[float, Dict]:
        """
        Validate quality of generated embeddings
        Returns (quality_score, metrics)
        """
        if not embeddings or not chunk_texts:
            return 0.0, {'error': 'No embeddings or chunks to validate'}
            
        metrics = {}
        
        # Check embedding dimensions
        actual_dim = len(embeddings[0])
        metrics['embedding_dim'] = actual_dim
        metrics['expected_dim'] = EMBEDDING_DIMENSION
        metrics['dim_match'] = actual_dim == EMBEDDING_DIMENSION
        
        # Calculate embedding norms (magnitudes)
        norms = [np.linalg.norm(emb) for emb in embeddings]
        metrics['avg_norm'] = sum(norms) / len(norms)
        metrics['min_norm'] = min(norms)
        metrics['max_norm'] = max(norms)
        
        # Check for zero or near-zero embeddings (failed encodings)
        near_zero_count = sum(1 for norm in norms if norm < 0.1)
        metrics['near_zero_embeddings'] = near_zero_count
        
        # Semantic similarity between adjacent chunks (should be related but not identical)
        similarities = []
        if len(embeddings) > 1:
            for i in range(len(embeddings) - 1):
                dot_product = np.dot(embeddings[i], embeddings[i+1])
                sim = dot_product / (norms[i] * norms[i+1]) if norms[i] > 0 and norms[i+1] > 0 else 0
                similarities.append(sim)
                
            metrics['avg_adjacent_similarity'] = sum(similarities) / len(similarities)
            metrics['max_adjacent_similarity'] = max(similarities)
            metrics['min_adjacent_similarity'] = min(similarities)
            
        # Check for duplicate or near-duplicate embeddings
        duplicate_pairs = 0
        if len(embeddings) > 1:
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dot_product = np.dot(embeddings[i], embeddings[j])
                    sim = dot_product / (norms[i] * norms[j]) if norms[i] > 0 and norms[j] > 0 else 0
                    if sim > 0.98:  # Extremely similar embeddings
                        duplicate_pairs += 1
                        
        metrics['duplicate_embedding_pairs'] = duplicate_pairs
        
        # Calculate quality score
        score = 1.0  # Start with perfect score
        
        # Deductions for problems
        if not metrics['dim_match']:
            score -= 0.5
            
        # Deduct for near-zero embeddings
        score -= (near_zero_count / len(embeddings)) * 0.5 if len(embeddings) > 0 else 0
        
        # Deduct for duplicates (should be semantically different)
        score -= (duplicate_pairs / len(embeddings)) * 0.3 if len(embeddings) > 0 else 0
        
        # Check if adjacent chunks have reasonable similarity (not too high, not too low)
        if 'avg_adjacent_similarity' in metrics:
            avg_sim = metrics['avg_adjacent_similarity']
            if avg_sim < 0.1 or avg_sim > 0.95:
                score -= 0.2
                
        return max(0.0, score), metrics

# =====================================================================================
# PDF PROCESSING FUNCTIONS
# =====================================================================================

def locate_pdf(**context) -> str:
    """
    Locate the PDF file in S3 and download it locally.
    Returns the path to the downloaded PDF.
    """
    metrics.start_stage("pdf_location")

    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Check if file exists in S3
    s3_pdf_key = f"books/{os.path.basename(BOOK_LOCAL_PATH)}"  # Using basename from the config
    
    if s3_hook.check_for_key(s3_pdf_key, bucket_name=S3_BUCKET):
        logging.info(f"✅ PDF found in S3 at s3://{S3_BUCKET}/{s3_pdf_key}")
        
        # Create temporary directory to download the file
        temp_dir = tempfile.mkdtemp(prefix="book_pdf_")
        local_pdf_path = os.path.join(temp_dir, os.path.basename(BOOK_LOCAL_PATH))
        
        # Download the file
        s3_hook.get_key(s3_pdf_key, bucket_name=S3_BUCKET).download_file(local_pdf_path)
        logging.info(f"✅ PDF downloaded to {local_pdf_path}")
        
        # Store temp directory for later cleanup
        context['ti'].xcom_push(key='temp_dir', value=temp_dir)
        context['ti'].xcom_push(key='pdf_path', value=local_pdf_path)
        
        metrics.end_stage("pdf_location", status="success")
        return local_pdf_path
    else:
        error_msg = f"❌ PDF not found in S3: s3://{S3_BUCKET}/{s3_pdf_key}"
        metrics.error("pdf_location", error_msg)
        metrics.end_stage("pdf_location", status="failed")
        raise AirflowFailException(error_msg)

def process_pdf_with_mistral(**context) -> dict:
    """
    Step: Use 'process_pdf' from mistralparsing_userpdf.py to convert local PDF to markdown.
    """
    metrics.start_stage("pdf_processing")
    ti = context['ti']
    
    pdf_path = ti.xcom_pull(key='pdf_path', task_ids='locate_pdf')
    if not pdf_path or not os.path.exists(pdf_path):
        error_msg = f"PDF path not found: {pdf_path}"
        metrics.error("pdf_processing", error_msg)
        metrics.end_stage("pdf_processing", status="failed")
        raise AirflowFailException(error_msg)

    output_dir = tempfile.mkdtemp(prefix="mistral_pdf_")
    ti.xcom_push(key='markdown_output_dir', value=output_dir)

    try:
        start_time = time.time()
        # ✅ Call your custom function from 'mistralparsing_userpdf.py'
        markdown_path = process_pdf(
            pdf_path=Path(pdf_path),
            output_dir=Path(output_dir)
        )
        duration = time.time() - start_time
        
        # (Optional) track how many chars we got
        if os.path.exists(markdown_path):
            char_count = len(Path(markdown_path).read_text(encoding='utf-8'))
        else:
            char_count = 0
        
        # Log + metrics
        logging.info(f"Mistral PDF parse completed: {markdown_path} (chars={char_count})")
        metrics.record_page_metrics(
            page_num=0,  # dummy
            extraction_time=duration,
            char_count=char_count,
            validation_score=1.0
        )
        
        # XCom push path for the next step
        ti.xcom_push(key='mistral_markdown_path', value=markdown_path)

        metrics.end_stage("pdf_processing", status="success", metrics={
            "duration": duration, "char_count": char_count
        })
        return {"markdown_path": markdown_path}
    
    except Exception as e:
        error_msg = f"Error in process_pdf_with_mistral: {e}"
        metrics.error("pdf_processing", error_msg, e)
        metrics.end_stage("pdf_processing", status="failed")
        raise AirflowFailException(error_msg)
    
# =====================================================================================
# CHUNKING FUNCTIONS
# =====================================================================================

class SmartChunker:
    """
    Advanced text chunking with semantic boundary detection and debugging.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug_info = None
    
    def split_text(self, text):
        """Split text into semantically coherent chunks"""
        self.debug_info = {
            'input_length': len(text),
            'chunk_params': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            'boundary_scores': [],
            'boundary_types': []
        }
        
        chunks = []
        
        # First try to split by markdown headers
        header_splits = self._split_by_headers(text)
        
        # Process each section separately
        for section in header_splits:
            # Check if section exceeds chunk size limit
            if len(section) <= self.chunk_size:
                chunks.append(section)
                self.debug_info['boundary_types'].append('header')
            else:
                # For longer sections, split by paragraphs
                paragraph_chunks = self._split_by_paragraphs(section)
                chunks.extend(paragraph_chunks)
                self.debug_info['boundary_types'].extend(['paragraph'] * len(paragraph_chunks))
        
        # Apply chunk overlap
        if self.chunk_overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # Add overlap from previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                    overlapped_chunks.append(overlap_text + chunks[i])
            chunks = overlapped_chunks
        
        # Final quality check - ensure no chunk exceeds maximum size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Force split at maximum size
                sub_chunks = self._force_split_chunk(chunk)
                final_chunks.extend(sub_chunks)
                self.debug_info['boundary_types'].extend(['forced'] * (len(sub_chunks) - 1))
        
        # Update debug info
        self.debug_info.update({
            'chunk_count': len(final_chunks),
            'avg_chunk_size': sum(len(c) for c in final_chunks) / max(1, len(final_chunks)),
            'min_chunk_size': min((len(c) for c in final_chunks), default=0),
            'max_chunk_size': max((len(c) for c in final_chunks), default=0),
        })
        
        return final_chunks
    
    def _split_by_headers(self, text):
        """Split text at markdown headers for semantic boundaries"""
        import re
        header_pattern = re.compile(r'^(#{1,3}\s+[^\n]+)$', re.MULTILINE)
        
        # Find all headers in the text
        headers = list(header_pattern.finditer(text))
        
        # If no headers, return the whole text as one section
        if not headers:
            return [text]
        
        # Split the text based on header positions
        sections = []
        start_pos = 0
        
        for match in headers:
            # If this is not the first header and there's content before it
            if match.start() > start_pos:
                sections.append(text[start_pos:match.start()])
            
            # Get header position for next slice
            start_pos = match.start()
        
        # Add the final section after the last header
        if start_pos < len(text):
            sections.append(text[start_pos:])
        
        return sections
    
    def _split_by_paragraphs(self, text):
        """Split text by paragraphs while respecting chunk size limits"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Handle case where a single paragraph exceeds chunk size
            if len(paragraph) > self.chunk_size:
                # First add any accumulated content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Then split this big paragraph further
                para_chunks = self._force_split_chunk(paragraph)
                chunks.extend(para_chunks)
                continue
            
            # If adding this paragraph would exceed chunk size, start a new chunk
            if current_chunk and (len(current_chunk) + len(paragraph) + 2 > self.chunk_size):
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _force_split_chunk(self, text):
        """Force split text at sentence boundaries or other safe points"""
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position (preferring sentence boundaries)
            end = start + self.chunk_size
            
            if end >= len(text):
                # If we're at the end of text, just add remainder
                chunks.append(text[start:])
                break
            
            # Try to find a sentence boundary before the limit
            # Look for period, question mark, or exclamation followed by space or newline
            boundary = -1
            for i in range(end - 1, start, -1):
                if i < len(text) and i > start and text[i-1] in '.!?' and (i == len(text) or text[i].isspace()):
                    boundary = i
                    break
            
            # If no sentence boundary, try to find a space
            if boundary == -1:
                for i in range(end - 1, start, -1):
                    if text[i].isspace():
                        boundary = i
                        break
            
            # If we found no good boundary, just split at the limit
            if boundary == -1 or boundary <= start:
                boundary = min(end, len(text))
            
            # Add this chunk and continue
            chunks.append(text[start:boundary])
            start = boundary
            
            # Remove leading whitespace from next chunk
            while start < len(text) and text[start].isspace():
                start += 1
        
        return chunks
    
    def get_debug_info(self):
        """Return debugging information about the chunking process"""
        return self.debug_info

def store_chunks_s3(chunks, s3_hook, book_name):
    """Store full text chunks in S3 for later retrieval"""
    
    # Create a dictionary mapping chunk_ids to full text
    chunks_dict = {
        f"kotler_chunk_{i}": {
            "text": chunk,
            "length": len(chunk),
            "timestamp": datetime.now().isoformat()
        }
        for i, chunk in enumerate(chunks)
    }
    
    # Create JSON payload
    chunks_payload = {
        "book_name": book_name,
        "created_at": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "chunks": chunks_dict
    }
    
    # Save to S3
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    chunks_key = f"{S3_BASE_FOLDER}/chunks/{book_name.replace('.pdf', '')}_chunks.json"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(chunks_payload, f, indent=2)
        chunks_path = f.name
        
    s3_hook.load_file(
        filename=chunks_path,
        key=chunks_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    os.unlink(chunks_path)
    logging.info(f"Stored {len(chunks)} chunks in S3: s3://{S3_BUCKET}/{chunks_key}")
    
    return chunks_key
def estimate_tokens(text):
    """Estimate token count for OpenAI models - moderately conservative (3 chars per token)"""
    return len(text) // 3  # Balanced estimation that's still safely conservative

def process_chunks_and_embeddings(**context):
    """
    Process chunks and create embeddings using OpenAI ada-002.
    Uses balanced chunking to avoid token limits while maintaining efficiency.
    """
    metrics.start_stage("chunking_and_embedding")
    ti = context['ti']

    # Get markdown path from previous task
    markdown_path = ti.xcom_pull(key='mistral_markdown_path', task_ids='process_pdf_with_mistral')
    if not markdown_path or not os.path.exists(markdown_path):
        error_msg = f"Missing or invalid markdown path: {markdown_path}"
        metrics.error("chunking_and_embedding", error_msg)
        metrics.end_stage("chunking_and_embedding", status="failed")
        raise AirflowFailException(error_msg)

    try:
        # Read markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # Use KamradtModifiedChunker for text splitting - use moderate chunk size
        chunker = KamradtModifiedChunker(avg_chunk_size=300, min_chunk_size=100)  # More reasonable chunk size
        raw_chunks = chunker.split_text(markdown_text)
        logging.info(f"Created {len(raw_chunks)} initial chunks from markdown text")
        
        # Process chunks to ensure they don't exceed token limits
        MAX_TOKENS = 2500  # About 1/3 of the API limit (balanced approach)
        chunks = []
        
        for chunk_idx, chunk in enumerate(raw_chunks):
            # Estimate tokens in chunk
            estimated_tokens = estimate_tokens(chunk)
            
            if estimated_tokens <= MAX_TOKENS:
                chunks.append(chunk)
            else:
                # Split large chunk into smaller pieces - log a warning
                logging.warning(f"Splitting large chunk #{chunk_idx} with ~{estimated_tokens} tokens ({len(chunk)} chars)")
                
                # Split by paragraphs first if possible
                paragraphs = chunk.split('\n\n')
                
                if len(paragraphs) > 1:
                    current_chunk = ""
                    for para in paragraphs:
                        para_tokens = estimate_tokens(para)
                        
                        # If a single paragraph is too large, force split it
                        if para_tokens > MAX_TOKENS:
                            # Add any accumulated content first
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = ""
                                
                            # Split paragraph by sentences
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            current_sent = ""
                            
                            for sentence in sentences:
                                sent_tokens = estimate_tokens(current_sent + sentence)
                                if sent_tokens > MAX_TOKENS:
                                    if current_sent:
                                        chunks.append(current_sent)
                                        current_sent = sentence
                                    else:
                                        # Emergency case - split by fixed size if a sentence is too long
                                        char_limit = MAX_TOKENS * 3  # Based on our token estimation
                                        for i in range(0, len(sentence), char_limit):
                                            chunks.append(sentence[i:i+char_limit])
                                else:
                                    if current_sent:
                                        current_sent += " " + sentence
                                    else:
                                        current_sent = sentence
                            
                            if current_sent:
                                chunks.append(current_sent)
                        # Regular case - check if adding this paragraph exceeds our limit
                        elif estimate_tokens(current_chunk + para) > MAX_TOKENS:
                            chunks.append(current_chunk)
                            current_chunk = para
                        else:
                            if current_chunk:
                                current_chunk += "\n\n" + para
                            else:
                                current_chunk = para
                    
                    # Add any remaining content
                    if current_chunk:
                        chunks.append(current_chunk)
                else:
                    # Force split by sentences as there are no paragraphs
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if estimate_tokens(current_chunk + sentence) > MAX_TOKENS:
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = sentence
                            else:
                                # Emergency split if sentence is too long
                                char_limit = MAX_TOKENS * 3
                                for i in range(0, len(sentence), char_limit):
                                    chunks.append(sentence[i:i+char_limit])
                        else:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                    
                    if current_chunk:
                        chunks.append(current_chunk)
        
        logging.info(f"After token limit processing: {len(chunks)} chunks")
        
        # Store full chunks in S3 before embedding
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        chunks_s3_key = store_chunks_s3(chunks, s3_hook, BOOK_NAME)
        logging.info(f"Stored full chunks in S3: s3://{S3_BUCKET}/{chunks_s3_key}")
        ti.xcom_push(key='chunks_s3_key', value=chunks_s3_key)
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except TypeError as e:
            if 'proxies' in str(e):
                # Handle case where proxies are being passed from environment
                import httpx
                # Create client without proxies
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    http_client=httpx.Client()
                )
            else:
                raise
        
        # Process chunks in small batches to avoid rate limits but be faster than one-by-one
        embeddings = []
        batch_size = 5  # Small batch size to balance speed and reliability
        
        logging.info(f"Generating OpenAI embeddings for {len(chunks)} chunks...")
        for i in range(0, len(chunks), batch_size):
            # Get batch of chunks
            batch = chunks[i:i+batch_size]
            
            # Add delay between batches to avoid rate limits
            if i > 0:
                time.sleep(1)
            
            # Final safety check for each chunk in the batch
            safe_batch = []
            for chunk in batch:
                if len(chunk) > 7500:  # About 2500 tokens with our estimation
                    safe_batch.append(chunk[:7500])  # Truncate if necessary
                else:
                    safe_batch.append(chunk)
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=safe_batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                if i % 50 == 0:  # Log progress every 50 batches
                    logging.info(f"Embedded chunks {i} to {i+len(batch)} of {len(chunks)}")
            except Exception as e:
                error_msg = f"Error generating embeddings for batch {i}: {str(e)}"
                metrics.error("embedding_generation", error_msg, e)
                
                # Fall back to individual processing for this batch
                for j, chunk in enumerate(safe_batch):
                    try:
                        time.sleep(2)  # Longer delay after an error
                        response = client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=[chunk]
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as e2:
                        error_msg = f"Error on individual chunk {i+j}: {str(e2)}"
                        metrics.error("embedding_generation", error_msg, e2)
                        # Add placeholder embedding
                        embeddings.append([0.0] * 1536)
        
        # Initialize Pinecone client
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Log available indexes for debugging
        available_indexes = pc.list_indexes().names()
        logging.info(f"Available Pinecone indexes: {available_indexes}")
        
        # Verify index exists or create it
        index_name = PINECONE_INDEX_NAME
        if index_name not in available_indexes:
            logging.info(f"Pinecone index '{index_name}' not found. Creating it now...")
            try:
                # Create new index
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI ada-002 dimensions
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logging.info(f"✅ Created Pinecone index '{index_name}'")
                
                # Wait for index to be ready
                wait_start = time.time()
                ready = False
                while not ready and time.time() - wait_start < 300:  # 5 min timeout
                    try:
                        index_info = pc.describe_index(index_name)
                        ready = index_info.status.get('ready', False)
                        if ready:
                            break
                    except Exception as e:
                        logging.warning(f"Waiting for index readiness... ({str(e)})")
                    
                    logging.info("Waiting for index to be ready...")
                    time.sleep(10)
                
                if not ready:
                    raise TimeoutError("Index creation timed out")
                    
                logging.info(f"✅ Pinecone index '{index_name}' is ready")
            except Exception as e:
                error_msg = f"Failed to create Pinecone index '{index_name}': {str(e)}"
                metrics.error("chunking_and_embedding", error_msg, e)
                raise AirflowFailException(error_msg)

        # Get index and verify configuration
        index = pc.Index(index_name)
        index_info = pc.describe_index(index_name)
        
        # Log index information for debugging
        logging.info(f"Connected to Pinecone index: {index_name}")
        logging.info(f"Index info: dimension={index_info.dimension}, metric={index_info.metric}")
        
        if index_info.dimension != 1536:
            error_msg = f"Index has wrong dimension: {index_info.dimension} (expected 1536)"
            metrics.error("chunking_and_embedding", error_msg)
            raise AirflowFailException(error_msg)

        # Use book-kotler namespace
        namespace = "book-kotler"
        
        # Prepare vectors with metadata for upserting
        upserts = []
        for i, emb in enumerate(embeddings):
             upserts.append({
                "id": f"kotler_chunk_{i}",
                "values": emb,
                "metadata": {
                    "source": "Marketing Management - Kotler",
                    "chunk_idx": i,
                    "text_preview": chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i],
                    "chunk_length": len(chunks[i]),
                    "source_path": markdown_path,
                    "s3_chunks_key": chunks_s3_key,  
                    "embedding_model": "text-embedding-ada-002",
                    "processing_timestamp": datetime.now().isoformat()
                }
            })

        # Batch upload vectors to Pinecone
        batch_size = 100  # Pinecone batch size limit
        successful_upserts = 0
        for i in range(0, len(upserts), batch_size):
            batch = upserts[i:i+batch_size]
            try:
                # Upsert with namespace
                index.upsert(vectors=batch, namespace=namespace)
                successful_upserts += len(batch)
                logging.info(f"Upserted vectors {i} to {i+len(batch)} of {len(upserts)}")
            except Exception as e:
                error_msg = f"Error upserting batch {i}: {str(e)}"
                metrics.error("vector_upload", error_msg, e)
                # Add delay after errors
                time.sleep(3)

        # Record metrics and end stage
        metrics.end_stage("chunking_and_embedding", status="success", metrics={
            "chunk_count": len(chunks),
            "vectors_uploaded": successful_upserts,
            "embedding_dimension": 1536,
            "namespace": namespace,
            "index_name": index_name,
            "average_chunk_length": sum(len(c) for c in chunks) / len(chunks)
        })

        # Push results to XCom
        ti.xcom_push(key='chunking_results', value={
            "chunk_count": len(chunks),
            "vectors_uploaded": successful_upserts,
            "namespace": namespace,
            "index_name": index_name,
            "embedding_model": "text-embedding-ada-002",
            "status": "success"
        })

        return {
            "chunk_count": len(chunks),
            "vectors_uploaded": successful_upserts,
            "status": "success"
        }

    except Exception as e:
        error_msg = f"Error in process_chunks_and_embeddings: {str(e)}"
        metrics.error("chunking_and_embedding", error_msg, e)
        metrics.end_stage("chunking_and_embedding", status="failed")
        raise AirflowFailException(error_msg)

# =====================================================================================
# VALIDATION AND REPORTING
# =====================================================================================

def verify_and_report(**context):
    """
    Verify the pipeline results and generate a comprehensive report.
    Tests sample queries to validate the embedding quality.
    """
    metrics.start_stage("verification")
    
    # Get results from previous tasks
    pdf_results = context['ti'].xcom_pull(key='pdf_processing_results', task_ids='process_pdf_with_mistral')
    chunk_results = context['ti'].xcom_pull(key='chunking_results', task_ids='process_chunks_and_embeddings')
    
    # Initialize S3 hook for report upload
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Verify Pinecone integration
    try:
        # Updated Pinecone initialization using the new API
        from pinecone import Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_env = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists using the new API
        index_list = pc.list_indexes().names()
        if PINECONE_INDEX_NAME not in index_list:
            metrics.warning(f"Pinecone index {PINECONE_INDEX_NAME} not found")
            pinecone_info = {"status": "index_not_found"}
        else:
            # Connect to index using the new API
            index = pc.Index(PINECONE_INDEX_NAME)
            
            # Get index stats
            stats = index.describe_index_stats()
            pinecone_info = {
                "status": "available",
                "vector_count": stats['total_vector_count'],
                "namespaces": list(stats['namespaces'].keys()) if 'namespaces' in stats else [],
                "dimension": stats.get('dimension', EMBEDDING_DIMENSION),
                "namespace_counts": stats.get('namespaces', {})
            }
            
            # Try a test query with OpenAI embeddings
            try:
                # ✅ CHANGE: Use OpenAI instead of sentence-transformers
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Create a test query vector with OpenAI
                query_text = "marketing strategy"
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query_text]
                )
                query_vector = response.data[0].embedding
                
                # Query the index using the new API with namespace
                query_response = index.query(
                    vector=query_vector,
                    top_k=5,
                    include_metadata=True,
                    namespace=PINECONE_NAMESPACE
                )
                
                # Extract results
                pinecone_info["test_query"] = {
                    "query_text": query_text,
                    "match_count": len(query_response['matches']),
                    "namespace": PINECONE_NAMESPACE,
                    "sample_matches": [{
                        "id": m['id'],
                        "score": m['score'],
                        "metadata": m['metadata']
                    } for m in query_response['matches'][:3]]
                }
                
            except Exception as e:
                metrics.warning(f"Test query failed: {str(e)}")
                pinecone_info["test_query"] = {"status": "failed", "error": str(e)}
        
    except ImportError:
        pinecone_info = {"status": "import_failed"}
    except Exception as e:
        metrics.warning(f"Pinecone verification failed: {str(e)}")
        pinecone_info = {"status": "error", "message": str(e)}
    
    # Generate comprehensive report
    report = {
        "book_name": BOOK_NAME,
        "execution_date": context['execution_date'].isoformat(),
        "completion_time": datetime.now().isoformat(),
        "pdf_processing": pdf_results,
        "chunking": chunk_results,
        "pinecone": pinecone_info,
        "performance_metrics": metrics.get_summary()
    }
    
    # Create a markdown report for human reading
    md_report = f"""# Pipeline Execution Report: {BOOK_NAME}

## Summary
- **Execution Date**: {context['execution_date'].strftime('%Y-%m-%d %H:%M:%S')}
- **Completion Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Status**: {'✅ SUCCESS' if pinecone_info.get('status') == 'available' else '⚠️ PARTIAL SUCCESS'}

## Processing Statistics
- **PDF Pages**: {pdf_results.get('total_batches', 0) * BATCH_SIZE if pdf_results else 'Unknown'}
- **Markdown Batches**: {pdf_results.get('successful_batches', 0) if pdf_results else 0} / {pdf_results.get('total_batches', 0) if pdf_results else 0}
- **Chunks Created**: {chunk_results.get('total_chunks', 0) if chunk_results else 0}
- **Vectors Stored**: {pinecone_info.get('vector_count', 0)} in index '{PINECONE_INDEX_NAME}'

## Performance
- **Total Duration**: {(time.time() - metrics.start_time) / 60:.2f} minutes
- **Memory Peak**: {max([m for _, m in metrics.memory_samples]) if metrics.memory_samples else 0:.2f} MB
- **Processing Rate**: {60 * pdf_results.get('total_batches', 0) * BATCH_SIZE / (time.time() - metrics.start_time) if pdf_results else 0:.2f} pages/minute

## Vector Testing
- **Test Query**: "{pinecone_info.get('test_query', {}).get('query_text', 'Not tested')}"
- **Results Found**: {pinecone_info.get('test_query', {}).get('match_count', 0)}

## Errors and Warnings
- **Errors**: {len(metrics.errors)}
- **Warnings**: {len(metrics.warnings)}
"""

    # Add error details if any
    if metrics.errors:
        md_report += "\n## Error Details\n\n"
        for i, error in enumerate(metrics.errors[:10]):  # Show at most 10 errors
            md_report += f"### Error {i+1}: {error.get('stage', 'Unknown Stage')}\n"
            md_report += f"- **Message**: {error.get('error', 'Unknown error')}\n"
            md_report += f"- **Type**: {error.get('error_type', 'Unknown')}\n\n"
        
        if len(metrics.errors) > 10:
            md_report += f"\n... and {len(metrics.errors) - 10} more errors. See full report for details.\n"
    
    # Save reports to S3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON report
    json_report_key = f"{S3_DIAGNOSTIC_FOLDER}/pipeline_report_{timestamp}.json"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(report, f, indent=2, default=str)
        json_path = f.name
        
    s3_hook.load_file(
        filename=json_path,
        key=json_report_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    os.unlink(json_path)
    
    # Save markdown report
    md_report_key = f"{S3_DIAGNOSTIC_FOLDER}/pipeline_report_{timestamp}.md"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(md_report)
        md_path = f.name
        
    s3_hook.load_file(
        filename=md_path,
        key=md_report_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    os.unlink(md_path)
    
    # Save metrics to a detailed diagnostic file
    metrics_report_key = metrics.save_report(s3_hook, "pipeline_metrics")
    
    # Clean up any temp directories
    temp_dir = context['ti'].xcom_pull(key='temp_dir', task_ids='locate_pdf')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")
        
    markdown_dir = context['ti'].xcom_pull(key='markdown_output_dir', task_ids='process_pdf_with_mistral')
    if markdown_dir and os.path.exists(markdown_dir):
        shutil.rmtree(markdown_dir)
        logging.info(f"Cleaned up markdown directory: {markdown_dir}")
    
    logging.info(f"Pipeline execution complete. Reports available at:\n" +
               f"- JSON: s3://{S3_BUCKET}/{json_report_key}\n" +
               f"- Markdown: s3://{S3_BUCKET}/{md_report_key}\n" +
               f"- Metrics: s3://{S3_BUCKET}/{metrics_report_key}")
    
    metrics.end_stage("verification", status="success")
    
    return {
        "status": "success" if pinecone_info.get('status') == 'available' else "partial_success",
        "reports": {
            "json": f"s3://{S3_BUCKET}/{json_report_key}",
            "markdown": f"s3://{S3_BUCKET}/{md_report_key}",
            "metrics": f"s3://{S3_BUCKET}/{metrics_report_key}"
        },
        "vectors": pinecone_info.get('vector_count', 0)
    }
# =====================================================================================
# DAG DEFINITION AND TASK DEPENDENCIES
# =====================================================================================

# Create DAG
dag = DAG(
    "book_to_vector_pipeline",
    default_args=default_args,
    description="Process marketing book PDF to markdown and Pinecone vectors with advanced debugging",
    schedule_interval=None,  # Manual trigger
    catchup=False
)

locate_pdf_task = PythonOperator(
    task_id='locate_pdf',
    python_callable=locate_pdf,
    dag=dag
)

process_pdf_task = PythonOperator(
    task_id='process_pdf_with_mistral',
    python_callable=process_pdf_with_mistral,
    dag=dag
)

chunking_task = PythonOperator(
    task_id='process_chunks_and_embeddings',
    python_callable=process_chunks_and_embeddings,
    dag=dag
)

verification_task = PythonOperator(
    task_id='verify_and_report',
    python_callable=verify_and_report,
    dag=dag
)

locate_pdf_task >> process_pdf_task >> chunking_task >> verification_task
