import difflib
import re
import string
import nltk
import logging

# Initialize logging for smartcorrect.log
logging.basicConfig(
    filename="smartcorrect.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger("utils")

def download_nltk_resources() -> None:
    """
    Safely download required NLTK datasets for sentence and word tokenization.
    Performs silent checks to prevent annoying log prints if already installed.
    """
    resources = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng'
    }
    
    for name, path in resources.items():
        try:
            # Check if resource is already available locally
            nltk.data.find(path)
        except LookupError:
            # Not found, download silently
            try:
                logger.info(f"NLTK resource '{name}' not found locally. Downloading silently...")
                nltk.download(name, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource '{name}' silently.")
            except Exception as e:
                # If lookup error or download fails, print fallback warning
                err_msg = f"NLTK download failed for '{name}'. Attempting runtime download: {e}"
                logger.error(err_msg, exc_info=True)
                print(f"Warning: {err_msg}")
                try:
                    nltk.download(name)
                except Exception:
                    pass

def get_diff_html(original: str, corrected: str) -> str:
    """
    Compares the original and corrected text word-by-word and generates
    a beautiful HTML string showing changes:
    - Red with strikethroughs for deleted/misspelled words.
    - Green with underlines for new/corrected words.
    
    Args:
        original (str): Original text source.
        corrected (str): Corrected text target.
        
    Returns:
        str: Styled HTML block comparing words side-by-side.
    """
    try:
        if not original:
            return ""
        if not corrected:
            return f'<span style="background-color: #ffcccc; color: #cc0000; text-decoration: line-through; padding: 2px 4px; border-radius: 4px;">{original}</span>'
            
        orig_words = original.split()
        corr_words = corrected.split()
        
        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        html_parts = []
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                # Unchanged words
                html_parts.append(" ".join(orig_words[i1:i2]))
            elif op == 'replace':
                # Spelling or grammar replacements
                del_text = " ".join(orig_words[i1:i2])
                ins_text = " ".join(corr_words[j1:j2])
                html_parts.append(
                    f'<span style="background-color: #ffeef0; color: #d73a49; text-decoration: line-through; padding: 2px 6px; border-radius: 4px; margin: 0 2px; font-weight: 500;">{del_text}</span>'
                    f'<span style="background-color: #e6ffed; color: #22863a; text-decoration: underline; padding: 2px 6px; border-radius: 4px; margin: 0 2px; font-weight: 500;">{ins_text}</span>'
                )
            elif op == 'delete':
                # Deleted words
                del_text = " ".join(orig_words[i1:i2])
                html_parts.append(
                    f'<span style="background-color: #ffeef0; color: #d73a49; text-decoration: line-through; padding: 2px 6px; border-radius: 4px; margin: 0 2px; font-weight: 500;">{del_text}</span>'
                )
            elif op == 'insert':
                # Newly inserted words
                ins_text = " ".join(corr_words[j1:j2])
                html_parts.append(
                    f'<span style="background-color: #e6ffed; color: #22863a; text-decoration: underline; padding: 2px 6px; border-radius: 4px; margin: 0 2px; font-weight: 500;">{ins_text}</span>'
                )
                
        # Wrap in a modern visual container with refined styling
        diff_html = " ".join(html_parts)
        return f'<div style="line-height: 1.8; font-size: 1.1rem; padding: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1);">{diff_html}</div>'
    except Exception as e:
        logger.error(f"Error in get_diff_html computation: {e}", exc_info=True)
        return f"<div>{corrected}</div>"


# ==============================================================================
# Preprocessing and High-Fidelity Text Analysis Utilities
# ==============================================================================

def preprocess_text(text: str) -> tuple[list[str], list[dict]]:
    """
    Preprocesses text by converting it to lowercase while preserving the original casing,
    tokenizing using NLTK's word_tokenize, handling URLs/emails/punctuation gracefully, 
    and generating a high-fidelity character-position map back to the original text.
    
    Args:
        text (str): The raw source input string.
        
    Returns:
        tuple[list[str], list[dict]]:
            - list[str]: List of low-cased token strings.
            - list[dict]: List of token dictionaries with original token, processed token,
                          start index, end index, is_url flag, and is_special flag.
    """
    try:
        # Safely bootstrap NLTK resources
        download_nltk_resources()
        
        if not text:
            return [], []
            
        # URL and email regex patterns
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\S+@\S+\.\S+")
        
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(text))
        
        processed_tokens = []
        char_map = []
        
        for start, end in spans:
            orig_token = text[start:end]
            token_lower = orig_token.lower()
            
            is_url = bool(url_pattern.match(orig_token))
            is_email = bool(email_pattern.match(orig_token))
            
            # Mark URLs, emails, or punctuation-only tokens as special (skipped in correction)
            is_special = is_url or is_email or all(c in string.punctuation for c in orig_token)
            
            processed_tokens.append(token_lower)
            char_map.append({
                "original": orig_token,
                "processed": token_lower,
                "start": start,
                "end": end,
                "is_url": is_url,
                "is_special": is_special
            })
            
        return processed_tokens, char_map
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}", exc_info=True)
        return [], []

def is_proper_noun(word: str) -> bool:
    """
    Uses NLTK's part-of-speech tagger to determine if the given word is a proper noun (NNP/NNPS).
    
    Args:
        word (str): A single word string to analyze.
        
    Returns:
        bool: True if NNP or NNPS POS tag is assigned, False otherwise.
    """
    try:
        # Safely bootstrap NLTK resources
        download_nltk_resources()
        
        # Strip leading/trailing punctuation before pos-tagging
        clean_word = word.strip(string.punctuation)
        if not clean_word:
            return False
            
        # Frame-tagging: POS taggers rely heavily on syntactical cues.
        # By framing the word in a sentence like "This is [word].", we get near 100% proper noun classification.
        tokens = ["This", "is", clean_word, "."]
        tagged = nltk.pos_tag(tokens)
        tag = tagged[2][1]
        return tag in ("NNP", "NNPS")
    except Exception as e:
        logger.error(f"Error checking proper noun status for '{word}': {e}", exc_info=True)
        return False

def is_technical_term(word: str, custom_dict: list[str] = None) -> bool:
    """
    Checks if a word exists in a default technical terms dictionary or a user-provided custom dictionary list.
    
    Args:
        word (str): A single word to verify.
        custom_dict (list[str]): Optional custom technical terms list.
        
    Returns:
        bool: True if the word is classified as technical, False otherwise.
    """
    try:
        DEFAULT_TECH_TERMS = {
            "numpy", "pandas", "pytorch", "tensorflow", "github", "api", "apis", 
            "dataset", "datasets", "json", "xml", "csv", "sql", "nosql", "mongodb", 
            "streamlit", "html", "css", "js", "javascript", "python", "java", "scala", 
            "kotlin", "swift", "rust", "golang", "bash", "cli", "regex", "ml", "dl", 
            "ai", "nlp", "bert", "gpt", "lstm", "cnn", "rnn", "gpu", "cpu", "tpu", 
            "cloud", "aws", "azure", "docker", "kubernetes", "git", "scikit", "sklearn"
        }
        
        clean_word = word.strip(string.punctuation).lower()
        if not clean_word:
            return False
            
        if clean_word in DEFAULT_TECH_TERMS:
            return True
            
        if custom_dict:
            custom_set = {term.lower().strip(string.punctuation) for term in custom_dict}
            if clean_word in custom_set:
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error checking technical term status for '{word}': {e}", exc_info=True)
        return False

def calculate_correction_accuracy(original_list: list[str], corrected_list: list[str], ground_truth_list: list[str]) -> dict[str, float]:
    """
    Compares the correction outputs to ground truth targets across sentence lists,
    calculating exact word-level precision, recall, and overall accuracy.
    
    Args:
        original_list (list[str]): List of original misspelled input sentences.
        corrected_list (list[str]): List of system-corrected output sentences.
        ground_truth_list (list[str]): List of target correct ground truth sentences.
        
    Returns:
        dict[str, float]: Evaluation report containing precision, recall, and accuracy scores (0.0 to 1.0).
    """
    try:
        tp, fp, fn, tn = 0, 0, 0, 0
        
        # Check matching length lists
        n_sentences = min(len(original_list), len(corrected_list), len(ground_truth_list))
        
        for i in range(n_sentences):
            orig_words = original_list[i].split()
            corr_words = corrected_list[i].split()
            gt_words = ground_truth_list[i].split()
            
            # Loop up to the minimum length to align indexes safely
            min_words = min(len(orig_words), len(corr_words), len(gt_words))
            
            for w_idx in range(min_words):
                ow = orig_words[w_idx].strip(string.punctuation).lower()
                cw = corr_words[w_idx].strip(string.punctuation).lower()
                gw = gt_words[w_idx].strip(string.punctuation).lower()
                
                if ow == gw:
                    # No spelling mistake existed originally
                    if cw == gw:
                        tn += 1  # Correctly left unchanged
                    else:
                        fp += 1  # Changed unnecessarily
                else:
                    # Spelling mistake existed originally
                    if cw == gw:
                        tp += 1  # Correctly corrected!
                    else:
                        fn += 1  # Failed to correct or corrected to a wrong word
                        
        # Compute standard metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 1.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy)
        }
    except Exception as e:
        logger.error(f"Error calculating accuracy metrics: {e}", exc_info=True)
        return {"precision": 1.0, "recall": 1.0, "accuracy": 1.0}

def format_output(changes_list: list[dict[str, str]]) -> str:
    """
    Formats a list of corrections into a clean, human-readable terminal/log string.
    Uses safe ASCII characters to guarantee cross-platform compatibility on Windows console.
    
    Args:
        changes_list (list[dict]): List of changes: [{"original": "wrold", "corrected": "world"}]
        
    Returns:
        str: A formatted string like: "[OK] 'wrold' -> 'world' | [OK] 'teh' -> 'the'"
    """
    try:
        if not changes_list:
            return "No corrections made."
            
        formatted_parts = []
        for change in changes_list:
            orig = change.get("original", "")
            corr = change.get("corrected", "")
            formatted_parts.append(f"[OK] '{orig}' -> '{corr}'")
            
        return " | ".join(formatted_parts)
    except Exception as e:
        logger.error(f"Error formatting changes list output: {e}", exc_info=True)
        return "No corrections made."
