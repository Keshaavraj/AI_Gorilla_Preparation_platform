#!/usr/bin/env python3
"""
Script to balance option lengths in quiz questions to prevent guessing based on length.
This makes incorrect answers more detailed and similar in length to correct answers.
"""

import json
import re
from collections import defaultdict

# Common padding phrases to make options more detailed
PADDING_VARIATIONS = [
    # For technical explanations
    ("reduces", "reduces by minimizing"),
    ("increases", "increases significantly"),
    ("improves", "improves overall"),
    ("decreases", "decreases substantially"),
    ("prevents", "prevents effectively"),
    ("enables", "enables better"),
    ("optimizes", "optimizes the"),
    ("handles", "handles efficiently"),
    ("processes", "processes correctly"),
    ("calculates", "calculates accurately"),

    # For yes/no statements
    ("always", "always in all cases"),
    ("never", "never under any circumstances"),
    ("can't", "cannot be used to"),
    ("doesn't", "does not"),
    ("won't", "will not"),

    # For technical terms - add elaboration
    ("overfitting", "overfitting to training data"),
    ("underfitting", "underfitting the data patterns"),
    ("bias", "bias in predictions"),
    ("variance", "variance in results"),
    ("accuracy", "overall accuracy"),
    ("precision", "precision of predictions"),
    ("recall", "recall of positive cases"),
]

def add_contextual_detail(option, target_length):
    """
    Add contextual detail to an option to reach target length.

    Args:
        option: Original option text
        target_length: Desired length

    Returns:
        Modified option with added detail
    """
    current_length = len(option)

    if current_length >= target_length * 0.9:  # Within 10% is good enough
        return option

    # Add detail based on content type
    result = option

    # Strategy 1: Expand common technical terms
    for short, long in PADDING_VARIATIONS:
        if short in result.lower() and len(result) < target_length:
            # Only replace if it makes sense contextually
            result = re.sub(r'\b' + re.escape(short) + r'\b', long, result, count=1, flags=re.IGNORECASE)

    # Strategy 2: Add clarifying phrases
    if len(result) < target_length * 0.85:
        if result.endswith("data"):
            result += " points"
        elif result.endswith("model"):
            result += " architecture"
        elif result.endswith("algorithm"):
            result += " approach"
        elif result.endswith("method"):
            result += " implementation"
        elif "learning" in result.lower() and not result.endswith("learning"):
            result = result.replace("learning", "learning approach", 1)

    # Strategy 3: Add technical qualifiers for very short options
    if len(result) < target_length * 0.7:
        if "is" in result or "are" in result:
            if "not" not in result.lower():
                # Add technical detail
                if "correct" in result.lower():
                    result = result.replace("correct", "correct in this context", 1)
                elif "accurate" in result.lower():
                    result = result.replace("accurate", "accurate for this use case", 1)
                elif "appropriate" in result.lower():
                    result = result.replace("appropriate", "appropriate for this scenario", 1)

    # Strategy 4: Expand abbreviations and add specificity
    if len(result) < target_length * 0.8:
        replacements = {
            r'\bML\b': 'Machine Learning',
            r'\bAI\b': 'Artificial Intelligence',
            r'\bNN\b': 'Neural Network',
            r'\bCNN\b': 'Convolutional Neural Network',
            r'\bRNN\b': 'Recurrent Neural Network',
            r'\bAPI\b': 'API interface',
            r'\bDB\b': 'database',
        }
        for pattern, replacement in replacements.items():
            if len(result) < target_length:
                result = re.sub(pattern, replacement, result)

    return result

def balance_question_lengths(question, target_ratio=0.85):
    """
    Balance the lengths of options in a question.

    Args:
        question: Question dictionary
        target_ratio: Target ratio of min to max option length (default 0.85)

    Returns:
        Modified question
    """
    options = question['options']
    lengths = [len(opt) for opt in options]
    max_length = max(lengths)

    # Calculate target length for shorter options
    target_length = int(max_length * target_ratio)

    # Balance each option
    balanced_options = []
    for i, option in enumerate(options):
        if len(option) < target_length:
            balanced = add_contextual_detail(option, target_length)
            balanced_options.append(balanced)
        else:
            balanced_options.append(option)

    question['options'] = balanced_options
    return question

def analyze_length_distribution(data):
    """Analyze and display option length statistics."""
    correct_lengths = []
    incorrect_lengths = []

    for category, questions in data['categories'].items():
        for q in questions:
            correct_idx = q['correct_answer']
            for i, opt in enumerate(q['options']):
                if i == correct_idx:
                    correct_lengths.append(len(opt))
                else:
                    incorrect_lengths.append(len(opt))

    avg_correct = sum(correct_lengths) / len(correct_lengths)
    avg_incorrect = sum(incorrect_lengths) / len(incorrect_lengths)

    print(f"\nOption Length Statistics:")
    print(f"  Correct answers:   avg={avg_correct:5.1f}, min={min(correct_lengths):3d}, max={max(correct_lengths):3d}")
    print(f"  Incorrect answers: avg={avg_incorrect:5.1f}, min={min(incorrect_lengths):3d}, max={max(incorrect_lengths):3d}")
    print(f"  Difference: {abs(avg_correct - avg_incorrect):.1f} characters")

    return avg_correct, avg_incorrect

def balance_all_questions(input_file, output_file=None):
    """
    Balance option lengths for all questions.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input)
    """
    # Load database
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=== BEFORE BALANCING ===")
    analyze_length_distribution(data)

    # Balance all questions
    questions_processed = 0
    for category, questions in data['categories'].items():
        for i, question in enumerate(questions):
            questions[i] = balance_question_lengths(question)
            questions_processed += 1

    print(f"\n✅ Balanced {questions_processed} questions")

    print("\n=== AFTER BALANCING ===")
    analyze_length_distribution(data)

    # Save
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_path}")

if __name__ == "__main__":
    input_file = "questions_db.json"

    print("⚖️  Quiz Option Length Balancer")
    print("=" * 50)

    # Note: We already have a backup from the previous script
    print("📦 Using existing backup from shuffle operation")

    # Balance questions
    balance_all_questions(input_file)

    print("\n" + "=" * 50)
    print("✨ Done! Options are now more uniform in length.")
    print("This makes it much harder to guess based on length!")
