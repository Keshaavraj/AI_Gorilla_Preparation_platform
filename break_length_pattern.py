#!/usr/bin/env python3
"""
Break the length pattern by making some incorrect answers longer than correct answers.
This prevents the "pick the longest answer" strategy from working.
"""

import json
import random

# Plausible-sounding technical additions that make incorrect answers longer
TECHNICAL_ELABORATIONS = [
    ", which optimizes the computational complexity",
    ", ensuring better convergence properties",
    ", while maintaining numerical stability",
    ", which improves the overall performance metrics",
    ", thereby reducing the computational overhead",
    ", leading to more efficient resource utilization",
    ", while preserving the mathematical properties",
    ", which enhances the model's generalization capability",
    ", ensuring compatibility with distributed systems",
    ", while maintaining backward compatibility",
    ", which provides better scalability characteristics",
    ", thereby improving the training efficiency",
    ", leading to faster convergence during optimization",
    ", while reducing the memory footprint significantly",
    ", which enables parallel processing capabilities",
]

def extend_option_plausibly(option):
    """
    Extend an option with plausible-sounding technical detail.

    Args:
        option: Original option text

    Returns:
        Extended option
    """
    # Don't extend if already very long
    if len(option) > 90:
        return option

    # Add elaboration based on what the option is about
    if any(word in option.lower() for word in ['algorithm', 'method', 'approach', 'technique']):
        additions = [
            " through iterative optimization",
            " using dynamic programming principles",
            " by leveraging statistical properties",
            " through gradient-based methods",
        ]
    elif any(word in option.lower() for word in ['model', 'network', 'system']):
        additions = [
            " with regularization techniques",
            " through architectural improvements",
            " by minimizing the loss function",
            " using ensemble methods",
        ]
    elif any(word in option.lower() for word in ['data', 'feature', 'input']):
        additions = [
            " across multiple dimensions",
            " through normalization and scaling",
            " by applying transformation techniques",
            " using feature engineering methods",
        ]
    elif any(word in option.lower() for word in ['reduce', 'increase', 'improve', 'optimize']):
        additions = TECHNICAL_ELABORATIONS
    else:
        additions = TECHNICAL_ELABORATIONS

    # Add punctuation if needed
    extended = option.rstrip('.')
    if random.random() < 0.7:  # 70% of the time, add an elaboration
        extended += random.choice(additions)

    return extended

def break_length_pattern(question):
    """
    Modify a question so the longest answer isn't always correct.

    Strategy:
    - In ~40% of questions, make at least one incorrect answer longer than correct
    - This breaks the "always pick longest" pattern

    Args:
        question: Question dictionary

    Returns:
        Modified question
    """
    correct_idx = question['correct_answer']
    options = question['options']

    lengths = [len(opt) for opt in options]
    correct_length = lengths[correct_idx]
    max_length = max(lengths)

    # If correct answer is already not the longest, we're good
    if correct_length < max_length - 10:
        return question

    # If correct answer is the longest (or tied for longest), extend some incorrect answers
    # Target: make 40% of questions have incorrect as longest
    if random.random() < 0.4:
        # Extend 1-2 incorrect answers to be longer than correct
        incorrect_indices = [i for i in range(len(options)) if i != correct_idx]
        num_to_extend = random.randint(1, min(2, len(incorrect_indices)))

        for idx in random.sample(incorrect_indices, num_to_extend):
            # Extend this incorrect option
            extended = extend_option_plausibly(options[idx])

            # Keep extending until it's longer than correct answer
            attempts = 0
            while len(extended) <= correct_length + 5 and attempts < 3:
                extended = extend_option_plausibly(extended)
                attempts += 1

            options[idx] = extended

    question['options'] = options
    return question

def analyze_longest_is_correct(data):
    """
    Analyze how often the longest option is the correct answer.

    Returns:
        Percentage of questions where longest = correct
    """
    longest_is_correct = 0
    total = 0

    for category, questions in data['categories'].items():
        for q in questions:
            correct_idx = q['correct_answer']
            lengths = [len(opt) for opt in q['options']]
            max_length = max(lengths)

            if lengths[correct_idx] == max_length:
                longest_is_correct += 1
            total += 1

    percentage = (longest_is_correct / total * 100) if total > 0 else 0
    return longest_is_correct, total, percentage

def process_database(input_file, output_file=None):
    """Process the entire database to break length patterns."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=== BEFORE PATTERN BREAKING ===")
    longest_correct, total, pct = analyze_longest_is_correct(data)
    print(f"Longest option is correct: {longest_correct}/{total} ({pct:.1f}%)")
    print(f"⚠️  A 'pick longest' strategy would get {pct:.1f}% correct!")

    # Process all questions
    questions_processed = 0
    for category, questions in data['categories'].items():
        for i, question in enumerate(questions):
            questions[i] = break_length_pattern(question)
            questions_processed += 1

    print(f"\n✅ Processed {questions_processed} questions")

    print("\n=== AFTER PATTERN BREAKING ===")
    longest_correct, total, pct = analyze_longest_is_correct(data)
    print(f"Longest option is correct: {longest_correct}/{total} ({pct:.1f}%)")

    if pct < 40:
        print(f"✅ Excellent! 'Pick longest' strategy now only gets {pct:.1f}% correct")
        print(f"   (Should be ~25% for random guessing)")
    elif pct < 60:
        print(f"✅ Good! Pattern is significantly broken")
    else:
        print(f"⚠️  Pattern still exists, may need another pass")

    # Save
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_path}")

if __name__ == "__main__":
    input_file = "questions_db.json"

    print("🔨 Length Pattern Breaker")
    print("=" * 50)

    # Set random seed for reproducibility
    random.seed(42)

    process_database(input_file)

    print("\n" + "=" * 50)
    print("✨ Done! The longest answer is no longer always correct!")
