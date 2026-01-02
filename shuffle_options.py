#!/usr/bin/env python3
"""
Script to randomize quiz options and make correct answers less predictable.
This will shuffle the option positions for each question while maintaining correctness.
"""

import json
import random
from pathlib import Path

def shuffle_question_options(question):
    """
    Shuffle options for a single question and update correct_answer index.

    Args:
        question: Dictionary containing question data

    Returns:
        Modified question with shuffled options
    """
    # Get the correct answer text before shuffling
    correct_answer_idx = question['correct_answer']
    correct_answer_text = question['options'][correct_answer_idx]

    # Create a list of (index, option) pairs and shuffle it
    options_with_indices = list(enumerate(question['options']))
    random.shuffle(options_with_indices)

    # Extract shuffled options and find new position of correct answer
    shuffled_options = [opt for idx, opt in options_with_indices]
    new_correct_idx = next(i for i, (orig_idx, opt) in enumerate(options_with_indices)
                          if orig_idx == correct_answer_idx)

    # Update question
    question['options'] = shuffled_options
    question['correct_answer'] = new_correct_idx

    return question

def analyze_distribution(data):
    """Analyze and print the distribution of correct answers."""
    correct_answer_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0

    for category, questions in data['categories'].items():
        for question in questions:
            correct_answer_counts[question['correct_answer']] += 1
            total += 1

    print("\nCorrect Answer Distribution:")
    print(f"Total questions: {total}")
    for option_idx, count in sorted(correct_answer_counts.items()):
        option_letter = chr(65 + option_idx)  # A, B, C, D
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  Option {option_letter} (index {option_idx}): {count:3d} ({percentage:5.1f}%)")

def shuffle_all_questions(input_file, output_file=None, seed=None):
    """
    Shuffle all questions in the database.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input)
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)

    # Load the questions database
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=== BEFORE SHUFFLING ===")
    analyze_distribution(data)

    # Shuffle options for all questions in all categories
    questions_processed = 0
    for category, questions in data['categories'].items():
        for i, question in enumerate(questions):
            questions[i] = shuffle_question_options(question)
            questions_processed += 1

    print(f"\n✅ Shuffled {questions_processed} questions")

    print("\n=== AFTER SHUFFLING ===")
    analyze_distribution(data)

    # Save the modified database
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_path}")

    return data

def create_backup(file_path):
    """Create a backup of the original file."""
    from datetime import datetime
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(file_path, 'r') as src:
        with open(backup_path, 'w') as dst:
            dst.write(src.read())
    print(f"📦 Backup created: {backup_path}")
    return backup_path

if __name__ == "__main__":
    input_file = "questions_db.json"

    print("🎲 Quiz Options Shuffler")
    print("=" * 50)

    # Create backup
    create_backup(input_file)

    # Shuffle questions
    shuffle_all_questions(input_file)

    print("\n" + "=" * 50)
    print("✨ Done! Your quiz options are now randomized.")
    print("The correct answers are now evenly distributed.")
