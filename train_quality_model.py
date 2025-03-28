#!/usr/bin/env python3
"""
Script to train the quality classifier model.
"""

import argparse
import logging
import os
import time
from cs336data.quality_classifier import train_model

def setup_logging(log_file="quality_training.log", verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="Train quality classifier model")
    parser.add_argument("--training-file", type=str, default="quality_train.txt",
                       help="Path to training file")
    parser.add_argument("--model-output", type=str, default="quality_classifier.bin",
                       help="Path to save the trained model")
    parser.add_argument("--log-file", type=str, default="quality_training.log",
                       help="Path to log file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file, args.verbose)
    
    # Log system info
    logger.info("=" * 60)
    logger.info("Quality classifier training started")
    logger.info(f"Training file: {args.training_file}")
    logger.info(f"Model output: {args.model_output}")
    
    # Check if training file exists
    if not os.path.exists(args.training_file):
        logger.error(f"Training file not found: {args.training_file}")
        return
    
    # Count lines in training file
    num_lines = 0
    num_high = 0
    num_low = 0
    try:
        with open(args.training_file, 'r', encoding='utf-8') as f:
            for line in f:
                num_lines += 1
                if "__label__high" in line:
                    num_high += 1
                elif "__label__low" in line:
                    num_low += 1
        
        logger.info(f"Training file stats:")
        logger.info(f"  Total examples: {num_lines}")
        logger.info(f"  High-quality examples: {num_high}")
        logger.info(f"  Low-quality examples: {num_low}")
    except Exception as e:
        logger.error(f"Error reading training file: {e}", exc_info=True)
    
    # Train the model
    start_time = time.time()
    try:
        classifier = train_model(args.training_file, args.model_output, logger)
        end_time = time.time()
        
        # Log success
        logger.info(f"Model training completed successfully in {end_time - start_time:.2f} seconds")
        logger.info(f"Model saved to {args.model_output}")
        
        # Check file size
        if os.path.exists(args.model_output):
            size_mb = os.path.getsize(args.model_output) / (1024 * 1024)
            logger.info(f"Model file size: {size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        end_time = time.time()
        logger.info(f"Training failed after {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()