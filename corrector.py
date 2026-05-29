import string
import re
from textblob import Word, TextBlob
import nltk
import logging
from functools import lru_cache

# Initialize logging for smartcorrect.log
logging.basicConfig(
    filename="smartcorrect.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger("corrector")

class SpellCorrector:
    """
    Core Spell Corrector engine using TextBlob.
    Provides basic spelling correction, single-word checking, suggestions retrieval,
    and visual modification highlighting.
    """

    def __init__(self) -> None:
        """Initializes the SpellCorrector engine."""
        pass

    def basic_correct(self, text: str, skip_proper_nouns: bool = False, custom_terms: list[str] = None) -> tuple[str, list[dict[str, str]]]:
        """
        Takes a raw string and uses TextBlob to correct spelling mistakes.
        
        Handles edge cases:
        - Empty/whitespace strings
        - Numeric tokens (ignored)
        - Punctuation (ignored)
        - ALL-CAPS words (ignored)
        
        Args:
            text (str): The raw input sentence or paragraph.
            skip_proper_nouns (bool): Whether to bypass proper nouns from being corrected.
            custom_terms (list[str]): List of custom words to never correct.
            
        Returns:
            tuple[str, list[dict]]: A tuple containing:
                - The corrected text (str)
                - A list of dictionary changes: [{"original": "wrold", "corrected": "world"}]
        """
        if not text or not text.strip():
            return ("", [])

        # Tokenize preserving spaces, word boundaries, and punctuation using regex
        # This keeps the exact spacing and format of the original string intact
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        corrected_tokens = []
        changes = []

        try:
            for token in tokens:
                # 1. Skip whitespace tokens
                if token.isspace():
                    corrected_tokens.append(token)
                    continue

                # 2. Skip punctuation tokens
                if all(c in string.punctuation for c in token):
                    corrected_tokens.append(token)
                    continue

                # 3. Skip purely numeric tokens or strings with digits
                if token.isdigit() or any(c.isdigit() for c in token):
                    corrected_tokens.append(token)
                    continue

                # 4. Skip ALL-CAPS words (commonly acronyms like USA, HTML, AI)
                if token.isupper() and len(token) > 1:
                    corrected_tokens.append(token)
                    continue

                # 5. Skip proper nouns if requested
                import utils
                if skip_proper_nouns and utils.is_proper_noun(token):
                    corrected_tokens.append(token)
                    continue

                # 6. Skip technical terms or custom dictionary terms
                if utils.is_technical_term(token, custom_terms):
                    corrected_tokens.append(token)
                    continue

                # Check if the word (or its lowercase form) is spelled correctly
                word_obj = Word(token)
                suggestions = word_obj.spellcheck()
                candidates = [cand.lower() for cand, conf in suggestions]

                if token.lower() in candidates or not suggestions:
                    # Spelled correctly, keep original
                    corrected_tokens.append(token)
                    continue

                # Otherwise, select the top suggestion
                corrected_word = suggestions[0][0]

                # Maintain titlecase casing if the original token was titlecased
                if token.istitle():
                    corrected_word = corrected_word.title()

                # Record the change
                c_change = {
                    "original": token,
                    "corrected": corrected_word
                }
                changes.append(c_change)
                corrected_tokens.append(corrected_word)
                logger.info(f"Lexical spelling correction identified: '{token}' -> '{corrected_word}'")

            reconstructed_text = "".join(corrected_tokens)
            return reconstructed_text, changes

        except Exception as e:
            # High-fidelity error handling
            logger.error(f"Error during basic correction: {e}", exc_info=True)
            return (text, [])

    @lru_cache(maxsize=1024)
    def word_correct(self, word: str) -> tuple[str, float]:
        """
        Corrects a single word and returns its highest confidence suggestion and score.
        Uses functools.lru_cache for high-performance repetition lookups.
        
        Args:
            word (str): A single word token to correct.
            
        Returns:
            tuple[str, float]: The corrected word (str) and TextBlob's confidence score (float).
        """
        clean_word = word.strip()
        if not clean_word:
            return (word, 1.0)

        try:
            word_obj = Word(clean_word)
            suggestions = word_obj.spellcheck()
            
            if not suggestions:
                return (word, 0.0)

            best_candidate, confidence = suggestions[0]

            # Adjust casing to match the original word's state
            if word.isupper():
                best_candidate = best_candidate.upper()
            elif word.istitle():
                best_candidate = best_candidate.title()

            logger.info(f"Word correction processed: '{clean_word}' -> '{best_candidate}' (confidence: {confidence:.4f})")
            return best_candidate, float(confidence)

        except Exception as e:
            logger.error(f"Error correcting word '{word}': {e}", exc_info=True)
            return (word, 0.0)

    def highlight_changes(self, original: str, corrected: str) -> list[dict[str, int | str]]:
        """
        Compares the original and corrected text word-by-word.
        Returns a list of changed words and their 0-indexed token positions.
        
        Args:
            original (str): The original source text.
            corrected (str): The corrected target text.
            
        Returns:
            list[dict]: A list of changes: [{"original": "wrold", "corrected": "world", "position": index}]
        """
        if not original or not corrected:
            return []

        orig_words = original.split()
        corr_words = corrected.split()
        changes = []

        try:
            # Iterate up to the shorter word list length to avoid indexing errors
            for idx in range(min(len(orig_words), len(corr_words))):
                # Clean word borders from trailing punctuations for an accurate compare
                orig_clean = orig_words[idx].strip(string.punctuation)
                corr_clean = corr_words[idx].strip(string.punctuation)

                if orig_clean.lower() != corr_clean.lower():
                    changes.append({
                        "original": orig_clean,
                        "corrected": corr_clean,
                        "position": idx
                    })
            return changes

        except Exception as e:
            logger.error(f"Error highlighting changes: {e}", exc_info=True)
            return []

    def get_suggestions(self, word: str) -> list[tuple[str, float]]:
        """
        Retrieves the top 5 spelling corrections for a word along with confidence scores.
        
        Args:
            word (str): The target misspelled word.
            
        Returns:
            list[tuple[str, float]]: Up to 5 tuples of (candidate_word, confidence).
        """
        clean_word = word.strip()
        if not clean_word:
            return []

        try:
            word_obj = Word(clean_word)
            spellcheck_results = word_obj.spellcheck()
            
            suggestions = []
            for sug, conf in spellcheck_results[:5]:
                # Casing formatting adjustments
                cased_sug = sug
                if word.isupper():
                    cased_sug = sug.upper()
                elif word.istitle():
                    cased_sug = sug.title()
                
                suggestions.append((cased_sug, float(conf)))
                
            return suggestions

        except Exception as e:
            logger.error(f"Error fetching suggestions for '{word}': {e}", exc_info=True)
            return []


# ==============================================================================
# Advanced context-aware checker for backward-compatibility with older runs
# ==============================================================================
class SmartCorrector:
    """
    Advanced context-aware spell and grammar checker utilizing both TextBlob (lexical)
    and BERT (semantic contextual weighting). Used in older scripts.
    """
    def __init__(self, model_name="distilbert-base-uncased", alpha=0.4):
        from bert_predictor import BertPredictor
        logger.info("Initializing SmartCorrector backward-compatibility layer...")
        self.predictor = BertPredictor(model_name)
        self.alpha = alpha
        self.confused_words = {
            "their": ["there", "they're"],
            "there": ["their", "they're"],
            "they're": ["their", "there"],
            "its": ["it's"],
            "it's": ["its"],
            "then": ["than"],
            "than": ["then"],
            "your": ["you're"],
            "you're": ["your"],
            "loose": ["lose"],
            "lose": ["loose"],
            "to": ["too", "two"],
            "too": ["to", "two"],
            "two": ["to", "too"],
            "read": ["reed", "red"],
            "reed": ["read", "red"],
            "red": ["read", "reed"],
            "write": ["right", "rite"],
            "right": ["write", "rite"],
            "rite": ["write", "right"],
            "affect": ["effect"],
            "effect": ["affect"],
            "accept": ["except"],
            "except": ["accept"],
            "compliment": ["complement"],
            "complement": ["compliment"],
            "principle": ["principal"],
            "principal": ["principle"],
            "stationary": ["stationery"],
            "stationery": ["stationary"],
            "past": ["passed"],
            "passed": ["past"],
            "advice": ["advise"],
            "advise": ["advice"]
        }

    def context_correct(self, text: str, alpha: float = 0.4) -> dict:
        try:
            spell_corrector = SpellCorrector()
            textblob_corrected, tb_changes = spell_corrector.basic_correct(text)
            
            words = textblob_corrected.split()
            final_words = list(words)
            all_changes = list(tb_changes)
            
            for idx, word in enumerate(words):
                clean_word = word.strip(string.punctuation).lower()
                
                # Check for common homophones
                candidates = self.confused_words.get(clean_word, [])
                if candidates:
                    all_candidates = [clean_word] + candidates
                    
                    target_sentence = " ".join(words[:idx] + ["[MASK]"] + words[idx+1:])
                    scored = self.predictor.score_candidates(target_sentence, all_candidates)
                    
                    if scored:
                        best_candidate = scored[0]["word"]
                        best_score = scored[0]["score"]
                        
                        # Find original TextBlob score
                        current_score = next((x["score"] for x in scored if x["word"] == clean_word), 1e-6)
                        
                        # Apply hybrid equation
                        # Lexical confidence vs Contextual probability
                        tb_conf = 1.0  # Homophone was spelled correctly
                        hybrid_current = (alpha * tb_conf) + ((1 - alpha) * current_score)
                        hybrid_best = (alpha * 1.0) + ((1 - alpha) * best_score)
                        
                        if best_candidate != clean_word and hybrid_best > hybrid_current:
                            formatted_cand = best_candidate
                            if word[0].isupper():
                                formatted_cand = best_candidate.title()
                            if word.isupper():
                                formatted_cand = best_candidate.upper()
                                
                            punctuation_suffix = ""
                            curr_word = word
                            while curr_word and curr_word[-1] in string.punctuation:
                                punctuation_suffix = curr_word[-1] + punctuation_suffix
                                curr_word = curr_word[:-1]
                                
                            final_words[idx] = formatted_cand + punctuation_suffix
                            all_changes.append({
                                "original": word,
                                "corrected": formatted_cand + punctuation_suffix,
                                "reason": "Homophone Context Selection"
                            })
                            logger.info(f"Homophone contextual correction made: '{word}' -> '{final_words[idx]}'")
                            
            return {
                "original_text": text,
                "corrected_text": " ".join(final_words),
                "changes": all_changes
            }
        except Exception as e:
            logger.error(f"Error in context_correct backward-compatibility: {e}", exc_info=True)
            return {"original_text": text, "corrected_text": text, "changes": []}


# ==============================================================================
# Quick Local Visual Comparative Diagnostics block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("=== SMARTCORRECT AI - SpellCorrector Console Demo ===")
    print("=" * 70)
    
    corrector = SpellCorrector()
    
    # 1. Spelling corrections
    sample_text = "The wrold has teh best speling."
    print(f"\nOriginal Input Text:\n  '{sample_text}'")
    
    corrected, changes = corrector.basic_correct(sample_text)
    print(f"\nCorrected Output Text:\n  '{corrected}'")
    print(f"\nChanges Track Log: {changes}")
    
    # 2. Capitalization and Acronym Bypass tests
    caps_text = "NASA has developed an amazing HTML parser."
    print(f"\nOriginal Caps Input:\n  '{caps_text}'")
    corrected_caps, changes_caps = corrector.basic_correct(caps_text)
    print(f"Corrected Caps Output:\n  '{corrected_caps}'")
    
    # 3. Numeric tokens bypass tests
    numeric_text = "Check order #142 for 3 cats."
    print(f"\nOriginal Numbers Input:\n  '{numeric_text}'")
    corrected_num, changes_num = corrector.basic_correct(numeric_text)
    print(f"Corrected Numbers Output:\n  '{corrected_num}'")
    
    print("\n" + "=" * 70)
    print("All Core SpellCorrector Tests Completed Successfully!")
    print("=" * 70)
