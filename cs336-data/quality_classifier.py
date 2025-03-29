"""
Quality classifier implementation using fastText.
"""

import os
import re
import time
import logging
from typing import Tuple, Optional, Dict, Any
import fasttext

class QualityClassifier:
    """
    A classifier that determines text quality using fastText.
    """
    
    def __init__(self, model_path: Optional[str] = None, logger=None):
        """
        Initialize the quality classifier.
        
        Args:
            model_path: Path to a pre-trained fastText model
            logger: Logger for status updates
        """
        self.model_path = model_path
        self.model = None
        self.lang_model = None
        
        if logger is None:
            self.logger = logging.getLogger("quality_classifier")
        else:
            self.logger = logger
        
        # Try to load quality classifier model
        if model_path and os.path.exists(model_path):
            try:
                self.logger.info(f"Loading quality model from {model_path}")
                start_time = time.time()
                self.model = fasttext.load_model(model_path)
                end_time = time.time()
                self.logger.info(f"Quality model loaded in {end_time - start_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error loading quality model: {e}", exc_info=True)
        else:
            if model_path:
                self.logger.warning(f"Quality model file not found: {model_path}")
            else:
                self.logger.info("No quality model path provided")
        
        # Try to load language identification model as fallback
        lang_model_path = "lid.176.bin"
        if os.path.exists(lang_model_path):
            try:
                self.logger.info(f"Loading language model from {lang_model_path}")
                start_time = time.time()
                self.lang_model = fasttext.load_model(lang_model_path)
                end_time = time.time()
                self.logger.info(f"Language model loaded in {end_time - start_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error loading language model: {e}", exc_info=True)
                self.logger.info("Using fallback classification")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, training_file: str, **kwargs) -> None:
        """
        Train the quality classifier.
        
        Args:
            training_file: Path to fastText training file
            **kwargs: Additional training parameters
        """
        # Default parameters
        params = {
            'lr': 0.1,
            'epoch': 25,
            'wordNgrams': 2,
            'dim': 100,
            'minCount': 2,
            'loss': 'softmax'
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        # Count training examples
        high_count = 0
        low_count = 0
        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "__label__high" in line:
                    high_count += 1
                elif "__label__low" in line:
                    low_count += 1
        
        self.logger.info(f"Training with {high_count} high-quality examples and {low_count} low-quality examples")
        self.logger.info(f"Training parameters: {params}")
        
        # Train model
        self.logger.info("Starting model training...")
        start_time = time.time()
        
        self.model = fasttext.train_supervised(
            training_file,
            lr=params['lr'],
            epoch=params['epoch'],
            wordNgrams=params['wordNgrams'],
            dim=params['dim'],
            minCount=params['minCount'],
            loss=params['loss']
        )
        
        end_time = time.time()
        self.logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")
        
        # Save model
        if self.model_path:
            self.logger.info(f"Saving model to {self.model_path}")
            self.model.save_model(self.model_path)
            self.logger.info("Model saved successfully")
    
    def fallback_classification(self, text: str) -> Tuple[str, float]:
        """
        Fallback classification using language model or heuristics.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (label, confidence)
        """
        lower_text = text.lower()
        
        # Try using language model if available
        if self.lang_model and len(text) > 20:
            try:
                # Use language model to help assess quality
                lang_pred = self.lang_model.predict(text)
                lang = lang_pred[0][0].replace('__label__', '')
                conf = float(lang_pred[1][0])
                
                # Check if text is primarily in English (en) with high confidence
                if lang == 'en' and conf > 0.8:
                    # Further heuristic checks for English text
                    word_count = len(text.split())
                    avg_word_len = sum(len(w) for w in text.split()) / max(1, word_count)
                    
                    # More sophisticated analysis for English text
                    if word_count > 100 and avg_word_len > 5:
                        # Longer text with complex words tends to be higher quality
                        return "high", 0.7
                    elif any(marker in lower_text for marker in ["study", "research", "analysis", "evidence", "conclusion", "findings"]):
                        return "high", 0.75
                    else:
                        return "low", 0.6
                else:
                    # Non-English or low-confidence text
                    return "low", 0.65
            except:
                # If language model fails, fall back to simple heuristics
                pass
        
        # Simple fallback heuristics when no models are available
        if not text or len(text) < 50:
            return "low", 0.9
        elif any(pattern in lower_text for pattern in ["!!!", "omg", "$$$", "click", "free", "buy now", "limited time", "amazing"]):
            return "low", 0.85
        elif any(pattern in lower_text for pattern in ["study", "research", "analysis", "according to", "evidence", "methodology"]):
            return "high", 0.8
        else:
            return "low", 0.6
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict quality label and confidence for a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (label, confidence)
        """
        if not self.model:
            # If quality model is not loaded, use fallback
            self.logger.debug("Using fallback classification (no quality model loaded)")
            return self.fallback_classification(text)
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            self.logger.debug("Empty text after preprocessing, classifying as low-quality")
            return "low", 1.0
            
        # Get predictions from fastText model
        predictions = self.model.predict(processed_text)
        label = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])
        
        self.logger.debug(f"Classification result: {label} with confidence {confidence:.4f}")
        return label, confidence
    
    def get_quality_score(self, text: str) -> float:
        """
        Get a numeric quality score for text.
        
        Args:
            text: Input text
            
        Returns:
            Quality score between 0 and 1 (higher is better quality)
        """
        label, confidence = self.predict(text)
        
        # Convert to numeric score
        # If high quality, return confidence
        # If low quality, return 1 - confidence
        score = confidence if label == "high" else 1 - confidence
        self.logger.debug(f"Quality score: {score:.4f} (label: {label}, confidence: {confidence:.4f})")
        return score

def train_model(training_file: str, model_path: str = "quality_classifier.bin", logger=None) -> QualityClassifier:
    """
    Train a quality classifier model.
    
    Args:
        training_file: Path to fastText training file
        model_path: Path to save the model
        logger: Logger for status updates
        
    Returns:
        Trained QualityClassifier
    """
    if logger is None:
        logger = logging.getLogger("quality_classifier.train")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    logger.info(f"Initializing quality classifier training from {training_file}")
    
    # Initialize the classifier
    classifier = QualityClassifier(model_path)
    
    # Train the model
    classifier.train(training_file)
    
    # Test the model on training data
    result = classifier.model.test(training_file)
    precision = result[1]
    recall = result[2]
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Model evaluation on training data:")
    logger.info(f"  Samples: {result[0]}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return classifier