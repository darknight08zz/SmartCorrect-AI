# How to run:
# python -m pytest test_smartcorrect.py -v
# or
# python test_smartcorrect.py

import unittest
import string
import sys
import pandas as pd

# Add the workspace directory to path if needed
sys.path.append(".")

from corrector import SpellCorrector
from bert_predictor import BERTContextCorrector
import utils

# ==============================================================================
# 1. UNIT TEST SUITE
# ==============================================================================

class TestSpellCorrector(unittest.TestCase):
    """Unit tests for SpellCorrector lexical checking logic."""

    def setUp(self) -> None:
        self.spell_engine = SpellCorrector()

    def test_basic_spelling(self) -> None:
        """Tests that common spelling mistakes are correctly replaced."""
        test_text = "The wrold has teh best speling."
        corrected, changes = self.spell_engine.basic_correct(test_text)
        
        self.assertIn("world", corrected.lower())
        self.assertIn("the", corrected.lower())
        self.assertIn("spelling", corrected.lower())
        self.assertTrue(len(changes) >= 3)

    def test_empty_input(self) -> None:
        """Ensures that empty or whitespace strings return empty strings without raising errors."""
        corrected_empty, changes_empty = self.spell_engine.basic_correct("")
        corrected_space, changes_space = self.spell_engine.basic_correct("   ")
        
        self.assertEqual(corrected_empty, "")
        self.assertEqual(changes_empty, [])
        self.assertEqual(corrected_space, "")
        self.assertEqual(changes_space, [])

    def test_numbers_unchanged(self) -> None:
        """Verifies that purely numeric tokens or token digits are left untouched."""
        test_text = "I have 3 cats and 12 dogs."
        corrected, changes = self.spell_engine.basic_correct(test_text)
        
        self.assertEqual(corrected, test_text)
        self.assertEqual(changes, [])

    def test_proper_nouns_skipped(self) -> None:
        """Ensures that proper nouns like Rahul are preserved when skip_proper_nouns is set."""
        test_text = "Rahul lives in Mumbai."
        corrected, changes = self.spell_engine.basic_correct(test_text, skip_proper_nouns=True)
        
        self.assertIn("Rahul", corrected)
        self.assertIn("Mumbai", corrected)
        self.assertEqual(changes, [])

    def test_all_caps_skipped(self) -> None:
        """Verifies that ALL-CAPS acronyms are skipped from correction."""
        test_text = "The NASA administrator reviewed the HTML code."
        corrected, changes = self.spell_engine.basic_correct(test_text)
        
        self.assertIn("NASA", corrected)
        self.assertIn("HTML", corrected)

    def test_suggestions_returned(self) -> None:
        """Verifies that word_correct and get_suggestions return valid predictions and scores."""
        word = "wrold"
        corrected_word, confidence = self.spell_engine.word_correct(word)
        suggestions = self.spell_engine.get_suggestions(word)
        
        self.assertEqual(corrected_word.lower(), "world")  # Case-insensitive validation
        self.assertTrue(confidence >= 0.0)
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0][0].lower(), "world")


class TestBERTCorrector(unittest.TestCase):
    """Unit tests for the deep BERTContextCorrector MLM logic."""

    def setUp(self) -> None:
        self.bert_engine = BERTContextCorrector()

    def test_context_correction(self) -> None:
        """Tests homophone and context correction using BERT MLM predictions."""
        if self.bert_engine.nlp is None:
            self.skipTest("BERT model is not loaded or system is offline.")
            
        sentence = "I went to the beech yesterday."
        # Use low threshold to trigger correction easily
        results = self.bert_engine.full_context_correct(sentence, threshold=0.001)
        corrected = results["corrected_text"]
        
        self.assertIn("beach", corrected.lower())
        self.assertTrue(len(results["context_changes"]) >= 1)

    def test_fallback_on_error(self) -> None:
        """Ensures the BERT corrector falls back to TextBlob gracefully when pipeline is None."""
        # Backup pipeline
        original_nlp = self.bert_engine.nlp
        self.bert_engine.nlp = None
        
        sentence = "The wrold is beautiful."
        results = self.bert_engine.full_context_correct(sentence)
        
        # Restore pipeline
        self.bert_engine.nlp = original_nlp
        
        self.assertIn("world", results["corrected_text"].lower())
        self.assertTrue(len(results["spelling_changes"]) >= 1)
        self.assertEqual(results["context_changes"], [])

    def test_confidence_score(self) -> None:
        """Verifies that BERT MLM returned confidence score falls between 0.0 and 1.0."""
        if self.bert_engine.nlp is None:
            self.skipTest("BERT model is not loaded or system is offline.")
            
        sentence = "She is a grate cook."
        results = self.bert_engine.full_context_correct(sentence, threshold=0.001)
        context_changes = results["context_changes"]
        
        if context_changes:
            score = context_changes[0]["confidence"]
            self.assertTrue(0.0 <= score <= 1.0)


class TestUtils(unittest.TestCase):
    """Unit tests for core preprocessors and metrics whitelists in utils.py."""

    def test_preprocess_returns_lowercase(self) -> None:
        """Ensures preprocess_text low-cases tokens and builds character spans mapping."""
        text = "Visit https://google.com for NLP!"
        tokens, char_map = utils.preprocess_text(text)
        
        self.assertIn("visit", tokens)
        self.assertTrue(any(t["is_special"] for t in char_map))
        self.assertEqual(char_map[0]["original"], "Visit")
        self.assertEqual(char_map[0]["processed"], "visit")

    def test_proper_noun_detection(self) -> None:
        """Validates proper noun framing checks."""
        self.assertTrue(utils.is_proper_noun("Google"))
        self.assertTrue(utils.is_proper_noun("Mumbai"))
        self.assertFalse(utils.is_proper_noun("running"))

    def test_technical_term_detection(self) -> None:
        """Validates that tech terms and custom whitellisted vocabulary lists are preserved."""
        self.assertTrue(utils.is_technical_term("numpy"))
        self.assertTrue(utils.is_technical_term("github"))
        self.assertTrue(utils.is_technical_term("API"))
        self.assertFalse(utils.is_technical_term("house"))
        
        # Custom term scan
        self.assertTrue(utils.is_technical_term("SmartCorrect", ["SmartCorrect"]))


# ==============================================================================
# 2. RUNTIME EVALUATION & BENCHMARK SYSTEM
# ==============================================================================

def run_benchmark() -> None:
    """Runs a 20-sentence spelling and semantic accuracy evaluation benchmark."""
    print("\n" + "=" * 80)
    print("=== SMARTCORRECT AI - 20-SENTENCE ACCURACY EVALUATION BENCHMARK ===")
    print("=" * 80)
    
    # Instantiate models
    bert_engine = BERTContextCorrector()
    
    # 20 Benchmark sentences with typos/homophones and their clean ground truths
    benchmark_cases = [
        ("The wrold is beautiful.", "The world is beautiful."),
        ("She is a grate cook.", "She is a great cook."),
        ("I went to the beech yesterday.", "I went to the beach yesterday."),
        ("I want to reed a book.", "I want to read a book."),
        ("This is a test sentence with no errors.", "This is a test sentence with no errors."),
        ("Teh dog barked loudly.", "The dog barked loudly."),
        ("I have 3 cats and 2 dogs.", "I have 3 cats and 2 dogs."),
        ("Rahul lives in Mumbai.", "Rahul lives in Mumbai."),
        ("NASA launched a new rocket.", "NASA launched a new rocket."),
        ("It is a loose-loose situation.", "It is a lose-lose situation."),
        ("Please check the API documentation.", "Please check the API documentation."),
        ("He will write a letter.", "He will write a letter."),
        ("I like your short.", "I like your story."),
        ("The weather is very beautifull.", "The weather is very beautiful."),
        ("Can you review my code on github?", "Can you review my code on github?"),
        ("This is the wrong way.", "This is the wrong way."),
        ("They went to there house.", "They went to there house."),
        ("We use numpy for matrix calculations.", "We use numpy for matrix calculations."),
        ("Speling mistakes are common.", "Spelling mistakes are common."),
        ("I have to two dogs.", "I have to two dogs.")
    ]
    
    print("\nEvaluating spelling and contextual semantics on benchmark dataset...")
    
    inputs = []
    expecteds = []
    gots = []
    correct_flags = []
    
    for idx, (inp, expected) in enumerate(benchmark_cases, 1):
        # Run hybrid corrector with skip proper nouns enabled
        # Set low confidence threshold so BERT handles all context cases
        res = bert_engine.full_context_correct(inp, skip_proper_nouns=True, threshold=0.001)
        got = res["corrected_text"]
        
        # Clean both for evaluation spacing comparisons
        got_clean = " ".join(got.split())
        exp_clean = " ".join(expected.split())
        
        is_correct = got_clean.lower() == exp_clean.lower()
        
        inputs.append(inp)
        expecteds.append(expected)
        gots.append(got)
        correct_flags.append("✅" if is_correct else "❌")
        
    # Render detailed evaluation summary table
    df_eval = pd.DataFrame({
        "Index": range(1, 21),
        "Input Sentence": inputs,
        "Expected Ground Truth": expecteds,
        "SmartCorrect Output": gots,
        "Status": correct_flags
    })
    
    # Adjust pandas column formatting for elegant printing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    
    print("\n--- BENCHMARK RESULTS TABLE ---")
    print(df_eval.to_string(index=False))
    
    # Compute accuracy percentage
    correct_count = correct_flags.count("✅")
    accuracy_pct = (correct_count / 20.0) * 100
    
    print("\n" + "-" * 80)
    print(f"📊 Accuracy Statistics: {correct_count} / 20 correct sentences")
    print(f"🏆 Final System Accuracy: {accuracy_pct:.2f}%")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    # 1. Programmatically trigger unittest collection
    print("\n" + "=" * 80)
    print("=== SMARTCORRECT AI - RUNNING INTEGRATED UNIT TESTS ===")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load all local unit test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpellCorrector))
    suite.addTests(loader.loadTestsFromTestCase(TestBERTCorrector))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    # 2. Run the Benchmark evaluation suite
    run_benchmark()
    
    # Exit with code matching test failure status
    sys.exit(0 if test_result.wasSuccessful() else 1)
