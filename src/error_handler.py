"""
Error Handler Component for PDF Outline Extractor

This module provides comprehensive error handling, graceful degradation,
and recovery strategies for PDF processing failures.
"""

import logging
import traceback
import time
import psutil
import threading
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

# Import signal only on Unix-like systems
if platform.system() != 'Windows':
    import signal
else:
    signal = None

from .models import HeadingInfo
from .json_generator import JSONGenerator


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling strategies."""
    FILE_ACCESS = "file_access"
    PDF_CORRUPTION = "pdf_corruption"
    MEMORY_LIMIT = "memory_limit"
    TIMEOUT = "timeout"
    EXTRACTION_FAILURE = "extraction_failure"
    JSON_GENERATION = "json_generation"
    PERMISSION_ERROR = "permission_error"
    DISK_SPACE = "disk_space"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Container for error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ProcessingTimeout(Exception):
    """Custom exception for processing timeouts."""
    pass


class MemoryLimitExceeded(Exception):
    """Custom exception for memory limit violations."""
    pass


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, 
                 max_memory_mb: float = 1024.0,
                 processing_timeout_seconds: float = 300.0,
                 enable_performance_monitoring: bool = True):
        """
        Initialize ErrorHandler with configuration.
        
        Args:
            max_memory_mb: Maximum memory usage in MB before triggering cleanup
            processing_timeout_seconds: Maximum time for processing a single PDF
            enable_performance_monitoring: Whether to monitor performance metrics
        """
        self.max_memory_mb = max_memory_mb
        self.processing_timeout_seconds = processing_timeout_seconds
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.recovery_attempts: Dict[str, int] = {}
        
        # Performance monitoring
        self.start_time: Optional[float] = None
        self.peak_memory_mb: float = 0.0
        self.processed_files: int = 0
        self.processing_times: List[float] = []
        self.memory_samples: List[float] = []
        self.timeout_count: int = 0
        self.memory_cleanup_count: int = 0
        
        # JSON generator for fallback outputs
        self.json_generator = JSONGenerator()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Timeout handler setup
        self._timeout_handler_active = False
    
    def handle_pdf_read_error(self, file_path: Path, exception: Exception) -> Dict[str, Any]:
        """
        Handle errors when reading PDF files with graceful fallback.
        
        Args:
            file_path: Path to the PDF file that failed
            exception: The exception that occurred
            
        Returns:
            Fallback JSON data structure
        """
        error_info = self._categorize_error(exception, str(file_path))
        self._log_error(error_info, f"PDF read error for {file_path}")
        
        # Determine recovery strategy based on error type
        if error_info.category == ErrorCategory.FILE_ACCESS:
            self.logger.warning(f"File access error for {file_path}: {exception}")
            return self._create_fallback_output(file_path.stem, "File access denied")
        
        elif error_info.category == ErrorCategory.PDF_CORRUPTION:
            self.logger.warning(f"Corrupted PDF detected: {file_path}: {exception}")
            return self._create_fallback_output(file_path.stem, "Corrupted PDF file")
        
        elif error_info.category == ErrorCategory.PERMISSION_ERROR:
            self.logger.error(f"Permission error for {file_path}: {exception}")
            return self._create_fallback_output(file_path.stem, "Permission denied")
        
        else:
            self.logger.error(f"Unknown PDF read error for {file_path}: {exception}")
            return self._create_fallback_output(file_path.stem, "Unknown error")
    
    def handle_extraction_error(self, error: Exception, context: str, 
                              file_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Handle errors during text/outline extraction with recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Context description of where the error occurred
            file_path: Optional path to the file being processed
            
        Returns:
            Recovery data if applicable, None if processing should continue
        """
        error_info = self._categorize_error(error, context)
        self._log_error(error_info, f"Extraction error in {context}")
        
        # Implement recovery strategies based on error category
        if error_info.category == ErrorCategory.MEMORY_LIMIT:
            self.logger.warning(f"Memory limit exceeded in {context}, triggering cleanup")
            self._cleanup_memory()
            return None  # Continue processing after cleanup
        
        elif error_info.category == ErrorCategory.TIMEOUT:
            self.logger.warning(f"Processing timeout in {context}")
            if file_path:
                return self._create_fallback_output(file_path.stem, "Processing timeout")
            return None
        
        elif error_info.category == ErrorCategory.EXTRACTION_FAILURE:
            self.logger.info(f"Extraction failed in {context}, using fallback methods")
            return None  # Let caller try alternative extraction methods
        
        else:
            self.logger.error(f"Unhandled extraction error in {context}: {error}")
            if file_path:
                return self._create_fallback_output(file_path.stem, "Extraction failed")
            return None
    
    def create_fallback_output(self, filename: str, title: str = "", 
                             error_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Create fallback JSON output for failed extractions.
        
        Args:
            filename: Name of the source file
            title: Optional title to include
            error_reason: Optional reason for fallback (for logging)
            
        Returns:
            Valid JSON structure with empty outline
        """
        if error_reason:
            self.logger.info(f"Creating fallback output for {filename}: {error_reason}")
        
        try:
            return self.json_generator.create_empty_output(title)
        except Exception as e:
            self.logger.error(f"Failed to create fallback output for {filename}: {e}")
            # Ultimate fallback - hardcoded structure
            return {"title": "", "outline": []}
    
    def _create_fallback_output(self, filename: str, title: str = "") -> Dict[str, Any]:
        """
        Internal method to create fallback JSON output.
        
        Args:
            filename: Name of the source file
            title: Title to include in output (should be empty for error cases)
            
        Returns:
            Valid empty JSON structure
        """
        try:
            # For error cases, always use empty title
            return self.json_generator.create_empty_output("")
        except Exception as e:
            self.logger.error(f"Failed to create fallback output for {filename}: {e}")
            # Ultimate fallback - hardcoded structure
            return {"title": "", "outline": []}
    
    @contextmanager
    def timeout_context(self, timeout_seconds: Optional[float] = None):
        """
        Context manager for processing timeouts (cross-platform).
        
        Args:
            timeout_seconds: Timeout in seconds, uses default if None
            
        Raises:
            ProcessingTimeout: If processing exceeds timeout
        """
        timeout = timeout_seconds or self.processing_timeout_seconds
        
        if signal and hasattr(signal, 'SIGALRM'):
            # Unix-like systems: use signal-based timeout
            def timeout_handler(signum, frame):
                raise ProcessingTimeout(f"Processing exceeded {timeout} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            self._timeout_handler_active = True
            
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                self._timeout_handler_active = False
        else:
            # Windows: use threading-based timeout
            timeout_occurred = threading.Event()
            
            def timeout_handler():
                timeout_occurred.wait(timeout)
                if not timeout_occurred.is_set():
                    # This will be checked in the main thread
                    pass
            
            timer_thread = threading.Thread(target=timeout_handler, daemon=True)
            timer_thread.start()
            
            start_time = time.time()
            try:
                yield
                # Signal that processing completed successfully
                timeout_occurred.set()
            finally:
                # Check if timeout occurred
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.record_timeout()
                    raise ProcessingTimeout(f"Processing exceeded {timeout} seconds")
    
    @contextmanager
    def memory_monitoring_context(self):
        """
        Context manager for memory usage monitoring.
        
        Raises:
            MemoryLimitExceeded: If memory usage exceeds limits
        """
        initial_memory = self._get_memory_usage_mb()
        
        try:
            yield
        finally:
            current_memory = self._get_memory_usage_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            
            if current_memory > self.max_memory_mb:
                self.logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
                self._cleanup_memory()
    
    def safe_process_pdf(self, process_func: Callable, file_path: Path, 
                        *args, **kwargs) -> Dict[str, Any]:
        """
        Safely execute PDF processing function with comprehensive error handling.
        
        Args:
            process_func: Function to execute for PDF processing
            file_path: Path to the PDF file being processed
            *args: Arguments to pass to process_func
            **kwargs: Keyword arguments to pass to process_func
            
        Returns:
            Processing result or fallback output
        """
        start_time = time.time()
        
        try:
            with self.timeout_context():
                with self.memory_monitoring_context():
                    result = process_func(file_path, *args, **kwargs)
                    
                    # Validate result
                    if not isinstance(result, dict):
                        raise ValueError("Processing function must return dictionary")
                    
                    processing_time = time.time() - start_time
                    self.logger.debug(f"Successfully processed {file_path} in {processing_time:.2f}s")
                    
                    return result
        
        except ProcessingTimeout as e:
            self.logger.warning(f"Processing timeout for {file_path}: {e}")
            return self.handle_extraction_error(e, "PDF processing", file_path)
        
        except MemoryLimitExceeded as e:
            self.logger.warning(f"Memory limit exceeded for {file_path}: {e}")
            return self.handle_extraction_error(e, "PDF processing", file_path)
        
        except Exception as e:
            self.logger.error(f"Unexpected error processing {file_path}: {e}")
            return self.handle_pdf_read_error(file_path, e)
    
    def log_performance_metrics(self, context: str = ""):
        """
        Log current performance metrics.
        
        Args:
            context: Optional context description
        """
        if not self.enable_performance_monitoring:
            return
        
        current_memory = self._get_memory_usage_mb()
        elapsed_time = time.time() - (self.start_time or time.time())
        
        metrics = {
            "context": context,
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory_mb,
            "elapsed_time_s": elapsed_time,
            "processed_files": self.processed_files,
            "avg_time_per_file": elapsed_time / max(1, self.processed_files)
        }
        
        self.logger.info(f"Performance metrics: {metrics}")
    
    def start_monitoring(self):
        """Start performance monitoring session."""
        self.start_time = time.time()
        self.peak_memory_mb = self._get_memory_usage_mb()
        self.processed_files = 0
        self.logger.info("Performance monitoring started")
    
    def increment_processed_files(self, processing_time: Optional[float] = None):
        """
        Increment the count of processed files and record processing time.
        
        Args:
            processing_time: Time taken to process the file in seconds
        """
        self.processed_files += 1
        if processing_time is not None:
            self.processing_times.append(processing_time)
    
    def record_memory_sample(self):
        """Record current memory usage sample."""
        if self.enable_performance_monitoring:
            current_memory = self._get_memory_usage_mb()
            self.memory_samples.append(current_memory)
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
    
    def record_timeout(self):
        """Record a timeout occurrence."""
        self.timeout_count += 1
    
    def record_memory_cleanup(self):
        """Record a memory cleanup occurrence."""
        self.memory_cleanup_count += 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of errors encountered during processing.
        
        Returns:
            Dictionary with error statistics and details
        """
        if not self.error_history:
            return {
                "total_errors": 0,
                "error_categories": {},
                "error_severities": {},
                "recent_errors": []
            }
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = []
        for error in self.error_history[-10:]:
            recent_errors.append({
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "timestamp": error.timestamp
            })
        
        return {
            "total_errors": len(self.error_history),
            "error_categories": category_counts,
            "error_severities": severity_counts,
            "recent_errors": recent_errors
        }
    
    def _categorize_error(self, exception: Exception, context: str) -> ErrorInfo:
        """
        Categorize an error based on its type and context.
        
        Args:
            exception: The exception to categorize
            context: Context where the error occurred
            
        Returns:
            ErrorInfo object with categorization
        """
        error_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Categorize based on exception type and message
        if isinstance(exception, FileNotFoundError):
            category = ErrorCategory.FILE_ACCESS
            severity = ErrorSeverity.MEDIUM
        
        elif isinstance(exception, PermissionError):
            category = ErrorCategory.PERMISSION_ERROR
            severity = ErrorSeverity.HIGH
        
        elif isinstance(exception, ProcessingTimeout):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        
        elif isinstance(exception, MemoryLimitExceeded):
            category = ErrorCategory.MEMORY_LIMIT
            severity = ErrorSeverity.HIGH
        
        elif "corrupted" in error_message or "invalid pdf" in error_message:
            category = ErrorCategory.PDF_CORRUPTION
            severity = ErrorSeverity.MEDIUM
        
        elif "disk" in error_message and "space" in error_message:
            category = ErrorCategory.DISK_SPACE
            severity = ErrorSeverity.HIGH
        
        elif "memory" in error_message or "out of memory" in error_message:
            category = ErrorCategory.MEMORY_LIMIT
            severity = ErrorSeverity.HIGH
        
        elif any(keyword in error_message for keyword in ["extract", "parse", "decode"]):
            category = ErrorCategory.EXTRACTION_FAILURE
            severity = ErrorSeverity.LOW
        
        elif "json" in error_message or "schema" in error_message:
            category = ErrorCategory.JSON_GENERATION
            severity = ErrorSeverity.MEDIUM
        
        else:
            category = ErrorCategory.UNKNOWN
            severity = ErrorSeverity.MEDIUM
        
        return ErrorInfo(
            category=category,
            severity=severity,
            message=str(exception),
            exception=exception,
            context={"context": context, "error_type": error_type}
        )
    
    def _log_error(self, error_info: ErrorInfo, message: str):
        """
        Log error with appropriate level based on severity.
        
        Args:
            error_info: Error information
            message: Log message
        """
        # Add to error history
        self.error_history.append(error_info)
        
        # Log with appropriate level
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"{message}: {error_info.message}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(f"{message}: {error_info.message}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"{message}: {error_info.message}")
        else:
            self.logger.info(f"{message}: {error_info.message}")
        
        # Log stack trace for debugging if exception is available
        if error_info.exception and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Stack trace for {message}:", exc_info=error_info.exception)
    
    def _get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0
    
    def _cleanup_memory(self):
        """
        Perform memory cleanup operations.
        """
        try:
            import gc
            gc.collect()
            self.record_memory_cleanup()
            self.logger.debug("Memory cleanup performed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def get_detailed_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics and statistics.
        
        Returns:
            Dictionary with comprehensive performance data
        """
        current_memory = self._get_memory_usage_mb()
        elapsed_time = time.time() - (self.start_time or time.time())
        
        # Calculate processing time statistics
        processing_stats = {}
        if self.processing_times:
            processing_stats = {
                "min_time": min(self.processing_times),
                "max_time": max(self.processing_times),
                "avg_time": sum(self.processing_times) / len(self.processing_times),
                "total_processing_time": sum(self.processing_times)
            }
        
        # Calculate memory statistics
        memory_stats = {}
        if self.memory_samples:
            memory_stats = {
                "min_memory": min(self.memory_samples),
                "max_memory": max(self.memory_samples),
                "avg_memory": sum(self.memory_samples) / len(self.memory_samples),
                "memory_samples_count": len(self.memory_samples)
            }
        
        return {
            "session_metrics": {
                "elapsed_time_s": elapsed_time,
                "processed_files": self.processed_files,
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.peak_memory_mb,
                "timeout_count": self.timeout_count,
                "memory_cleanup_count": self.memory_cleanup_count
            },
            "processing_time_stats": processing_stats,
            "memory_usage_stats": memory_stats,
            "performance_ratios": {
                "files_per_second": self.processed_files / max(elapsed_time, 0.001),
                "memory_efficiency": current_memory / max(self.processed_files, 1),
                "timeout_rate": self.timeout_count / max(self.processed_files, 1)
            }
        }
    
    def monitor_processing_performance(self, file_path: Path, processing_func: Callable, 
                                     *args, **kwargs) -> Dict[str, Any]:
        """
        Monitor performance during PDF processing with detailed metrics.
        
        Args:
            file_path: Path to the PDF being processed
            processing_func: Function to execute
            *args: Arguments for processing function
            **kwargs: Keyword arguments for processing function
            
        Returns:
            Processing result with performance metrics
        """
        start_time = time.time()
        start_memory = self._get_memory_usage_mb()
        
        # Record initial memory sample
        self.record_memory_sample()
        
        try:
            result = self.safe_process_pdf(processing_func, file_path, *args, **kwargs)
            
            # Calculate performance metrics
            end_time = time.time()
            end_memory = self._get_memory_usage_mb()
            processing_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record metrics
            self.increment_processed_files(processing_time)
            self.record_memory_sample()
            
            # Log performance if enabled
            if self.enable_performance_monitoring:
                self.logger.debug(
                    f"File: {file_path.name}, Time: {processing_time:.2f}s, "
                    f"Memory: {start_memory:.1f}→{end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)"
                )
            
            return result
            
        except Exception as e:
            # Record failed processing
            processing_time = time.time() - start_time
            self.increment_processed_files(processing_time)
            raise
    
    def run_stress_test(self, test_files: List[Path], max_concurrent: int = 1) -> Dict[str, Any]:
        """
        Run stress test with multiple PDF files to validate performance.
        
        Args:
            test_files: List of PDF files to process
            max_concurrent: Maximum concurrent processing (for future use)
            
        Returns:
            Stress test results and performance metrics
        """
        self.logger.info(f"Starting stress test with {len(test_files)} files")
        
        # Reset monitoring
        self.start_monitoring()
        
        stress_results = {
            "total_files": len(test_files),
            "successful_files": 0,
            "failed_files": 0,
            "timeout_files": 0,
            "memory_issues": 0,
            "processing_times": [],
            "memory_peaks": [],
            "errors": []
        }
        
        for i, file_path in enumerate(test_files):
            try:
                self.logger.info(f"Processing file {i+1}/{len(test_files)}: {file_path.name}")
                
                # Mock processing function for stress test
                def mock_process(path):
                    # Simulate processing time based on file size
                    try:
                        file_size = path.stat().st_size
                        # Simulate processing: 1MB = 0.1 seconds
                        processing_time = (file_size / (1024 * 1024)) * 0.1
                        time.sleep(min(processing_time, 2.0))  # Cap at 2 seconds for testing
                        return {"title": f"Test {path.stem}", "outline": []}
                    except Exception:
                        return {"title": "", "outline": []}
                
                start_time = time.time()
                result = self.monitor_processing_performance(file_path, mock_process)
                processing_time = time.time() - start_time
                
                stress_results["successful_files"] += 1
                stress_results["processing_times"].append(processing_time)
                stress_results["memory_peaks"].append(self._get_memory_usage_mb())
                
                # Check for performance issues
                if processing_time > self.processing_timeout_seconds:
                    stress_results["timeout_files"] += 1
                
                current_memory = self._get_memory_usage_mb()
                if current_memory > self.max_memory_mb:
                    stress_results["memory_issues"] += 1
                
                # Log progress every 10 files
                if (i + 1) % 10 == 0:
                    self.log_performance_metrics(f"Stress test progress: {i+1}/{len(test_files)}")
                
            except ProcessingTimeout:
                stress_results["failed_files"] += 1
                stress_results["timeout_files"] += 1
                stress_results["errors"].append(f"Timeout: {file_path.name}")
                
            except MemoryLimitExceeded:
                stress_results["failed_files"] += 1
                stress_results["memory_issues"] += 1
                stress_results["errors"].append(f"Memory limit: {file_path.name}")
                
            except Exception as e:
                stress_results["failed_files"] += 1
                stress_results["errors"].append(f"Error in {file_path.name}: {str(e)}")
        
        # Calculate final statistics
        if stress_results["processing_times"]:
            stress_results["avg_processing_time"] = sum(stress_results["processing_times"]) / len(stress_results["processing_times"])
            stress_results["max_processing_time"] = max(stress_results["processing_times"])
            stress_results["min_processing_time"] = min(stress_results["processing_times"])
        
        if stress_results["memory_peaks"]:
            stress_results["peak_memory_usage"] = max(stress_results["memory_peaks"])
            stress_results["avg_memory_usage"] = sum(stress_results["memory_peaks"]) / len(stress_results["memory_peaks"])
        
        stress_results["success_rate"] = stress_results["successful_files"] / max(len(test_files), 1)
        stress_results["performance_metrics"] = self.get_detailed_performance_metrics()
        
        self.logger.info(f"Stress test completed: {stress_results['successful_files']}/{len(test_files)} successful")
        
        return stress_results
    
    def optimize_performance_settings(self, target_files: int, target_time_per_file: float = 2.0) -> Dict[str, Any]:
        """
        Analyze performance and suggest optimal settings.
        
        Args:
            target_files: Expected number of files to process
            target_time_per_file: Target processing time per file in seconds
            
        Returns:
            Optimization recommendations
        """
        current_metrics = self.get_detailed_performance_metrics()
        
        recommendations = {
            "current_settings": {
                "max_memory_mb": self.max_memory_mb,
                "processing_timeout_seconds": self.processing_timeout_seconds
            },
            "recommended_settings": {},
            "performance_analysis": {},
            "warnings": []
        }
        
        # Analyze current performance
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            if avg_time > target_time_per_file:
                recommendations["warnings"].append(
                    f"Average processing time ({avg_time:.2f}s) exceeds target ({target_time_per_file}s)"
                )
                
                # Suggest timeout adjustment
                recommended_timeout = max(avg_time * 2, target_time_per_file * 3)
                recommendations["recommended_settings"]["processing_timeout_seconds"] = recommended_timeout
        
        # Memory optimization
        if self.memory_samples:
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
            peak_memory = max(self.memory_samples)
            
            if peak_memory > self.max_memory_mb * 0.8:
                recommendations["warnings"].append(
                    f"Peak memory usage ({peak_memory:.1f}MB) is close to limit ({self.max_memory_mb}MB)"
                )
                
                # Suggest memory limit adjustment
                recommended_memory = peak_memory * 1.5
                recommendations["recommended_settings"]["max_memory_mb"] = recommended_memory
        
        # Performance analysis
        recommendations["performance_analysis"] = {
            "timeout_rate": self.timeout_count / max(self.processed_files, 1),
            "memory_cleanup_rate": self.memory_cleanup_count / max(self.processed_files, 1),
            "estimated_total_time": target_files * (sum(self.processing_times) / max(len(self.processing_times), 1)) if self.processing_times else 0
        }
        
        return recommendations
    
    def validate_processing_environment(self) -> Dict[str, Any]:
        """
        Validate the processing environment and system resources.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "memory_available": True,
            "disk_space_available": True,
            "system_responsive": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check available memory
            memory_info = psutil.virtual_memory()
            available_memory_mb = memory_info.available / (1024 * 1024)
            
            if available_memory_mb < self.max_memory_mb * 2:
                validation_results["memory_available"] = False
                validation_results["errors"].append(
                    f"Insufficient memory: {available_memory_mb:.1f}MB available, "
                    f"need at least {self.max_memory_mb * 2:.1f}MB"
                )
            elif available_memory_mb < self.max_memory_mb * 4:
                validation_results["warnings"].append(
                    f"Low memory: {available_memory_mb:.1f}MB available"
                )
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            available_disk_mb = disk_usage.free / (1024 * 1024)
            
            if available_disk_mb < 100:  # Less than 100MB
                validation_results["disk_space_available"] = False
                validation_results["errors"].append(
                    f"Insufficient disk space: {available_disk_mb:.1f}MB available"
                )
            elif available_disk_mb < 500:  # Less than 500MB
                validation_results["warnings"].append(
                    f"Low disk space: {available_disk_mb:.1f}MB available"
                )
            
            # Check system load
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                validation_results["system_responsive"] = False
                validation_results["warnings"].append(
                    f"High CPU usage: {cpu_percent}%"
                )
        
        except Exception as e:
            validation_results["errors"].append(f"Environment validation failed: {e}")
        
        return validation_results