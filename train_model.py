#!/usr/bin/env python3
import sys
import os
import fasttext
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Training parameters
training_file = "quality_train.txt"
model_output = "quality_classifier.bin"

logger.info(f"Training model from {training_file}")
model = fasttext.train_supervised(
    training_file,
    lr=0.1,
    epoch=25,
    wordNgrams=2,
    dim=100,
    minCount=2,
    loss='softmax'
)

logger.info(f"Saving model to {model_output}")
model.save_model(model_output)

# Evaluate model
result = model.test(training_file)
precision = result[1]
recall = result[2]
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

logger.info(f"Model evaluation:")
logger.info(f"  Samples: {result[0]}")
logger.info(f"  Precision: {precision:.4f}")
logger.info(f"  Recall: {recall:.4f}")
logger.info(f"  F1 Score: {f1:.4f}")