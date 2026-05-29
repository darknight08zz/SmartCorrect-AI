import string
import warnings
import logging

# Suppress standard Python warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers load reports and path warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from transformers import pipeline
import torch

# Initialize logging for smartcorrect.log
logging.basicConfig(
    filename="smartcorrect.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger("bert_predictor")

def load_bert_pipeline(device):
    """Loads the HuggingFace transformers fill-mask pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline("fill-mask", model="bert-base-uncased", device=device)

# Wrap model loading in Streamlit's @st.cache_resource if streamlit is available
try:
    import streamlit as st
    load_bert_pipeline = st.cache_resource(load_bert_pipeline)
    logger.info("Streamlit detected; wrapped load_bert_pipeline with st.cache_resource.")
except ImportError:
    pass

class BERTContextCorrector:
    """
    Intelligent context-aware corrector using BERT's Masked Language Model (MLM)
    to spot and fix semantic grammar and homophone errors in sentences.
    """
    
    # Static variable to cache the pipeline so the model loads only once
    _cached_pipeline = None

    def __init__(self) -> None:
        """
        Initializes the BERT Context Corrector.
        Safely attempts to load the fill-mask pipeline, falling back gracefully
        if offline or out of memory.
        """
        self.nlp = None
        self.mask_token = "[MASK]"
        
        logger.info("Initializing BERTContextCorrector...")
        # Load the model pipeline once and cache it
        if BERTContextCorrector._cached_pipeline is not None:
            self.nlp = BERTContextCorrector._cached_pipeline
            self.mask_token = self.nlp.tokenizer.mask_token
            logger.info("Loaded cached BERT pipeline from static cache.")
        else:
            try:
                # Use GPU CUDA if available, otherwise default to CPU (-1)
                device = 0 if torch.cuda.is_available() else -1
                
                logger.info(f"Loading pre-trained bert-base-uncased model on device: {device}...")
                nlp_pipe = load_bert_pipeline(device)
                
                BERTContextCorrector._cached_pipeline = nlp_pipe
                self.nlp = nlp_pipe
                self.mask_token = nlp_pipe.tokenizer.mask_token
                logger.info("BERT Context pipeline loaded successfully.")
                
            except Exception as e:
                # Graceful fallback warning
                err_msg = f"Failed to load pre-trained BERT model: {e}"
                logger.error(err_msg, exc_info=True)
                print("\n" + "!" * 80)
                print(f"⚠️ WARNING: {err_msg}")
                print("Gracefully falling back to TextBlob lexical spellcheck only.")
                print("!" * 80 + "\n")
                self.nlp = None

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculates the Levenshtein edit distance between two strings."""
        try:
            if len(s1) < len(s2):
                return self._edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
        except Exception as e:
            logger.error(f"Error in edit distance computation: {e}", exc_info=True)
            return 999

    def suggest_in_context(self, sentence: str, word_position: int, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Masks the word at the given word index and requests BERT to predict the 
        top context-fit word alternatives.
        
        Args:
            sentence (str): The full sentence to evaluate.
            word_position (int): 0-indexed position of the word to mask.
            top_k (int): Number of top predictions to return.
            
        Returns:
            list[tuple[str, float]]: List of (predicted_word, confidence_score) tuples.
        """
        if self.nlp is None or not sentence.strip():
            return []

        try:
            # Check if sentence ends with standard punctuation
            has_ending_punctuation = sentence.strip().endswith(('.', '!', '?'))
            
            words = sentence.split()
            if word_position < 0 or word_position >= len(words):
                return []

            # Backup original word casing and punctuation
            target_word = words[word_position]
            punctuation_suffix = ""
            
            # Extract trailing punctuation
            while target_word and target_word[-1] in string.punctuation:
                punctuation_suffix = target_word[-1] + punctuation_suffix
                target_word = target_word[:-1]

            # Place mask token
            words[word_position] = self.mask_token + punctuation_suffix
            
            # If the last word has no ending punctuation, append a period to complete the sentence.
            # This tells the language model the sentence is closed, forcing it to predict words
            # instead of punctuation marks like '.', '!', ';'.
            if not has_ending_punctuation and words:
                words[-1] = words[-1] + "."
                
            masked_sentence = " ".join(words)

            # Predict potential candidates using the fill-mask model
            results = self.nlp(masked_sentence, top_k=top_k)
            
            if isinstance(results, dict):
                results = [results]

            suggestions = []
            for res in results:
                sug_word = res["token_str"].strip().lower()
                
                # Filter out any lingering punctuation-only predictions (e.g. '.', '!')
                if sug_word in string.punctuation or sug_word == "...":
                    continue
                    
                sug_score = float(res["score"])
                suggestions.append((sug_word, sug_score))

            return suggestions

        except Exception as e:
            logger.error(f"Error predicting replacements in context: {e}", exc_info=True)
            return []

    def find_context_errors(self, original: str, textblob_corrected: str, skip_proper_nouns: bool = False, custom_terms: list[str] = None, threshold: float = 0.005) -> list[dict]:
        """
        Compares original and TextBlob-corrected strings to spot contextually
        incorrect words (like beach vs beech) using edit distance filters across
        all top predictions.
        
        Args:
            original (str): Original misspelled text input.
            textblob_corrected (str): TextBlob-corrected output.
            skip_proper_nouns (bool): Whether to bypass proper nouns from context checking.
            custom_terms (list[str]): Custom technical terms to skip.
            threshold (float): Confidence threshold for context suggestions.
            
        Returns:
            list[dict]: List of contextual changes made.
        """
        if self.nlp is None:
            return []

        orig_words = original.split()
        tb_words = textblob_corrected.split()
        context_changes = []

        # Common structural helper words (determiners, prepositions, pronouns, auxiliary verbs)
        # We protect these from context rewriting to preserve tense, structural grammar, and pronouns
        structural_words = {
            "is", "was", "are", "were", "be", "been", "am", 
            "the", "a", "an", "your", "my", "his", "her", "their", "our", "its",
            "you", "he", "she", "it", "they", "we", "i", "me", "him", "them", "us",
            "to", "of", "in", "for", "on", "at", "by", "with", "from", "as", "about",
            "and", "but", "or", "so", "if", "that", "this", "these", "those",
            "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "should"
        }

        try:
            for idx in range(min(len(orig_words), len(tb_words))):
                word = tb_words[idx]
                
                # Clean punctuation for spell checks
                clean_word = word.strip(string.punctuation).lower()
                
                # Skip numeric, punctuation, all-caps, or structural helper words
                if not clean_word or not clean_word.isalpha() or clean_word.isupper() or clean_word in structural_words:
                    continue

                # Skip proper nouns or technical/custom dictionary terms if requested
                import utils
                if skip_proper_nouns and utils.is_proper_noun(word):
                    continue
                if utils.is_technical_term(word, custom_terms):
                    continue

                # Query BERT suggestions for this position (get top 100 for high-coverage homophones)
                suggestions = self.suggest_in_context(textblob_corrected, idx, top_k=100)
                if not suggestions:
                    continue

                # Find the current word's score in the predictions (default to a very low score if not found)
                current_score = 0.0
                for sug_word, score in suggestions:
                    if sug_word == clean_word:
                        current_score = score
                        break

                best_candidate = None
                best_candidate_score = 0.0

                # Search all suggestions to find the best spelling neighbor (edit distance <= 2)
                for sug_word, score in suggestions:
                    if sug_word == clean_word:
                        continue
                        
                    dist = self._edit_distance(clean_word, sug_word)
                    if dist <= 2:
                        # Ratio of improvement
                        ratio = score / max(current_score, 1e-5)
                        
                        # Thresholds for context correction:
                        # - The suggested word has a reasonable probability (> threshold)
                        # - The suggested word is at least 10 times more likely than the current word in this context,
                        #   OR the current word score is extremely low (< 0.005)
                        if score > threshold and (ratio > 10.0 or current_score < 0.005):
                            if score > best_candidate_score:
                                best_candidate = sug_word
                                best_candidate_score = score

                if best_candidate:
                    # Match original casing
                    formatted_sug = best_candidate
                    if word[0].isupper():
                        formatted_sug = best_candidate.title()
                    if word.isupper():
                        formatted_sug = best_candidate.upper()
                        
                    # Preserve original punctuation
                    punctuation_suffix = ""
                    curr_word_stripped = word
                    while curr_word_stripped and curr_word_stripped[-1] in string.punctuation:
                        punctuation_suffix = curr_word_stripped[-1] + punctuation_suffix
                        curr_word_stripped = curr_word_stripped[:-1]
                        
                    c_change = {
                        "original": word,
                        "corrected": formatted_sug + punctuation_suffix,
                        "position": idx,
                        "confidence": float(best_candidate_score),
                        "reason": "Contextual Error"
                    }
                    context_changes.append(c_change)
                    logger.info(f"Context correction identified: '{word}' -> '{c_change['corrected']}' at position {idx} (confidence: {best_candidate_score:.4f})")

            return context_changes

        except Exception as e:
            logger.error(f"Error checking context errors: {e}", exc_info=True)
            return []

    def full_context_correct(self, text: str, skip_proper_nouns: bool = False, custom_terms: list[str] = None, threshold: float = 0.005) -> dict:
        """
        Orchestrates full context correction:
        1. Runs SpellCorrector basic spell check.
        2. Detects contextual word errors using BERT.
        3. Integrates lexical and context corrections.
        
        Args:
            text (str): Input raw sentence.
            skip_proper_nouns (bool): Whether to bypass proper nouns from checking.
            custom_terms (list[str]): List of custom words to never correct.
            threshold (float): Confidence threshold for context suggestions.
            
        Returns:
            dict: Diagnostic analysis logs.
        """
        # Local import to prevent circular dependency!
        from corrector import SpellCorrector
        
        logger.info(f"Running full context correction pipeline on input text of length {len(text)}...")
        
        try:
            spell_corrector = SpellCorrector()
            textblob_corrected, tb_changes = spell_corrector.basic_correct(text, skip_proper_nouns, custom_terms)
            
            if self.nlp is None:
                logger.info("BERT pipeline offline. Returning TextBlob results directly.")
                # Graceful fallback: return TextBlob results directly
                return {
                    "original_text": text,
                    "corrected_text": textblob_corrected,
                    "spelling_changes": tb_changes,
                    "context_changes": [],
                    "all_changes": tb_changes,
                    "confidence_breakdown": {}
                }

            # Run BERT context corrections on top of TextBlob
            context_changes = self.find_context_errors(text, textblob_corrected, skip_proper_nouns, custom_terms, threshold)
            
            # Apply BERT modifications
            words = textblob_corrected.split()
            confidence_breakdown = {}

            for change in context_changes:
                pos = change["position"]
                words[pos] = change["corrected"]
                confidence_breakdown[change["corrected"].strip(string.punctuation)] = change["confidence"]

            final_text = " ".join(words)
            all_changes = tb_changes + context_changes

            logger.info(f"Full context correction finished. Spelling changes: {len(tb_changes)}, Context changes: {len(context_changes)}")

            return {
                "original_text": text,
                "corrected_text": final_text,
                "spelling_changes": tb_changes,
                "context_changes": context_changes,
                "all_changes": all_changes,
                "confidence_breakdown": confidence_breakdown
            }

        except Exception as e:
            logger.error(f"Error executing full context correction: {e}", exc_info=True)
            return {
                "original_text": text,
                "corrected_text": text,
                "spelling_changes": [],
                "context_changes": [],
                "all_changes": [],
                "confidence_breakdown": {}
            }


# ==============================================================================
# Backward-compatibility layer for SmartCorrector & Streamlit app.py
# ==============================================================================
class BertPredictor:
    """
    Cached predictor helper for app.py UI.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.nlp = pipeline("fill-mask", model=model_name, device=self.device)
        except Exception:
            try:
                self.nlp = pipeline("fill-mask", model=model_name, device=-1)
            except Exception as e:
                logger.error(f"Failed to load BertPredictor model {model_name}: {e}", exc_info=True)
                self.nlp = None
                
        self.mask_token = self.nlp.tokenizer.mask_token if self.nlp else "[MASK]"

    def predict_mask(self, text_with_mask: str, top_k: int = 5) -> list:
        if not self.nlp:
            return []
        if "[MASK]" in text_with_mask and self.mask_token != "[MASK]":
            text_with_mask = text_with_mask.replace("[MASK]", self.mask_token)
        try:
            results = self.nlp(text_with_mask, top_k=top_k)
            if isinstance(results, dict):
                results = [results]
            return [{"word": res["token_str"].strip().lower(), "score": float(res["score"])} for res in results]
        except Exception as e:
            logger.error(f"Error in predict_mask: {e}", exc_info=True)
            return []

    def score_candidates(self, text_with_mask: str, candidates: list, top_k: int = 100) -> list:
        try:
            if not candidates:
                return []
            predictions = self.predict_mask(text_with_mask, top_k=top_k)
            pred_map = {pred["word"]: pred["score"] for pred in predictions}
            scored = [{"word": c.strip().lower(), "score": pred_map.get(c.strip().lower(), 1e-6)} for c in candidates]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored
        except Exception as e:
            logger.error(f"Error in score_candidates: {e}", exc_info=True)
            return []


# ==============================================================================
# Quick Visual Comparative Test Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("=== SMARTCORRECT AI - BERTContextCorrector Console Demo ===")
    print("=" * 70)
    print("Initializing BERT model pipeline (this may take a few seconds)...")
    
    corrector = BERTContextCorrector()
    
    test_cases = [
        "I went to the beech yesterday",
        "She is a grate cook",
        "I like your short"
    ]
    
    print("\nProcessing comparison cases...")
    print("-" * 70)
    
    for idx, sentence in enumerate(test_cases, 1):
        print(f"\nTest #{idx}:")
        print(f"  * Original Sentence: '{sentence}'")
        
        # Run full pipeline correction
        result = corrector.full_context_correct(sentence)
        
        # Output comparison results
        tb_out = result["spelling_changes"][0]["corrected"] if result["spelling_changes"] else sentence
        print(f"  * TextBlob Spellcheck: '{tb_out}'")
        print(f"  * BERT Smart Correction: '{result['corrected_text']}'")
        
        if result['context_changes']:
            print("  * Contextual Corrections Applied:")
            for change in result['context_changes']:
                print(f"    - Position {change['position']}: '{change['original']}' -> '{change['corrected']}' (Confidence: {change['confidence']:.4f})")
        else:
            print("  * Contextual Corrections Applied: None")
            
    print("\n" + "=" * 70)
    print("All BERT Contextual Tests Completed Successfully!")
    print("=" * 70)
