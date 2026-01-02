#!/usr/bin/env python3
"""
Enhanced script to make incorrect answer options (distractors) more detailed and plausible.
This prevents length-based guessing by making all options similarly detailed.
"""

import json
import random

# Templates for expanding short options
EXPANSION_TEMPLATES = {
    # For short affirmative/negative answers
    'yes': [
        "Yes, this is the correct approach for this scenario",
        "Yes, this method is always recommended",
        "Yes, this is a standard best practice",
        "Yes, this approach will solve the problem effectively",
    ],
    'no': [
        "No, this approach would not be effective in this case",
        "No, this method is not suitable for this purpose",
        "No, this would lead to incorrect results",
        "No, this is not the recommended approach here",
    ],
    'true': [
        "True, this statement accurately describes the behavior",
        "True, this is a well-known property in this context",
        "True, this is correct according to standard practices",
    ],
    'false': [
        "False, this is a common misconception about the topic",
        "False, this statement does not accurately represent the behavior",
        "False, this contradicts the fundamental principles",
    ],
    'always': [
        "This is always true in all cases and scenarios",
        "This property always holds regardless of the conditions",
        "This approach always produces the expected results",
    ],
    'never': [
        "This is never correct under any circumstances",
        "This approach never yields the desired outcome",
        "This property never holds in practical applications",
    ],
}

def expand_short_option(option, question_text, target_length):
    """
    Expand a short option to make it more detailed and closer to target length.

    Args:
        option: Original short option
        question_text: The question text for context
        target_length: Target length to reach

    Returns:
        Expanded option
    """
    option_lower = option.lower().strip()

    # Check for simple yes/no/true/false patterns
    for keyword, templates in EXPANSION_TEMPLATES.items():
        if option_lower.startswith(keyword):
            # Use a template if option is very short
            if len(option) < 30:
                return random.choice(templates)

    # Add context-aware expansions
    if len(option) < target_length * 0.6:
        # Check question context for clues
        expansions = []

        if 'model' in question_text.lower():
            expansions = [
                ' which ensures optimal model performance',
                ' leading to better generalization',
                ' improving model accuracy significantly',
                ' ensuring robust predictions',
            ]
        elif 'data' in question_text.lower():
            expansions = [
                ' across the entire dataset',
                ' for all data points in the collection',
                ' ensuring data integrity throughout',
                ' maintaining data consistency',
            ]
        elif 'algorithm' in question_text.lower():
            expansions = [
                ' using the most efficient algorithm',
                ' through algorithmic optimization',
                ' by applying the appropriate algorithm',
                ' with optimal algorithmic complexity',
            ]
        elif 'feature' in question_text.lower():
            expansions = [
                ' considering all feature interactions',
                ' across multiple feature dimensions',
                ' by analyzing feature importance',
                ' through feature engineering',
            ]

        if expansions and len(option) + len(random.choice(expansions)) <= target_length * 1.1:
            return option + random.choice(expansions)

    # Add technical qualifiers for common patterns
    if len(option) < target_length * 0.7:
        if "the" in option.lower() and len(option.split()) <= 5:
            technical_additions = [
                " in this specific context",
                " for this particular use case",
                " under these conditions",
                " given these parameters",
                " in practical applications",
            ]
            if not any(addition.strip() in option.lower() for addition in technical_additions):
                return option + random.choice(technical_additions)

    return option

def make_options_uniform_length(question, length_variance=15):
    """
    Make all options in a question similar in length.

    Args:
        question: Question dictionary
        length_variance: Maximum acceptable variance in characters (default 15)

    Returns:
        Modified question
    """
    options = question['options']
    question_text = question['question']

    # Calculate statistics
    lengths = [len(opt) for opt in options]
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)

    # If all options are already similar, skip
    if max_length - min(lengths) <= length_variance * 2:
        return question

    # Target length: slightly below max to avoid making things too long
    target_length = int(max_length * 0.85)

    # Expand short options
    new_options = []
    for option in options:
        if len(option) < target_length - length_variance:
            expanded = expand_short_option(option, question_text, target_length)
            # If expansion didn't help much, add generic detail
            if len(expanded) < target_length * 0.7:
                if expanded.endswith('.'):
                    expanded = expanded[:-1] + ", which is important to consider"
                else:
                    expanded = expanded + " based on standard principles"
            new_options.append(expanded)
        else:
            new_options.append(option)

    question['options'] = new_options
    return question

def analyze_length_stats(data):
    """Analyze option length statistics."""
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    length_differences = []  # Per question

    for category, questions in data['categories'].items():
        for q in questions:
            correct_idx = q['correct_answer']
            q_lengths = []
            for i, opt in enumerate(q['options']):
                length = len(opt)
                all_lengths.append(length)
                q_lengths.append(length)
                if i == correct_idx:
                    correct_lengths.append(length)
                else:
                    incorrect_lengths.append(length)

            # Track length difference within each question
            length_differences.append(max(q_lengths) - min(q_lengths))

    avg_correct = sum(correct_lengths) / len(correct_lengths)
    avg_incorrect = sum(incorrect_lengths) / len(incorrect_lengths)
    avg_difference = sum(length_differences) / len(length_differences)

    print(f"\nOption Length Statistics:")
    print(f"  Correct answers:   avg={avg_correct:5.1f}, min={min(correct_lengths):3d}, max={max(correct_lengths):3d}")
    print(f"  Incorrect answers: avg={avg_incorrect:5.1f}, min={min(incorrect_lengths):3d}, max={max(incorrect_lengths):3d}")
    print(f"  Global difference: {abs(avg_correct - avg_incorrect):.1f} chars")
    print(f"  Avg within-question difference: {avg_difference:.1f} chars")

    return avg_correct, avg_incorrect, avg_difference

def enhance_all_distractors(input_file, output_file=None):
    """Enhance all distractor options in the database."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=== BEFORE ENHANCEMENT ===")
    analyze_length_stats(data)

    # Enhance all questions
    questions_processed = 0
    for category, questions in data['categories'].items():
        for i, question in enumerate(questions):
            questions[i] = make_options_uniform_length(question)
            questions_processed += 1

    print(f"\n✅ Enhanced {questions_processed} questions")

    print("\n=== AFTER ENHANCEMENT ===")
    analyze_length_stats(data)

    # Save
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_path}")

if __name__ == "__main__":
    input_file = "questions_db.json"

    print("🎯 Enhanced Distractor Generator")
    print("=" * 50)

    enhance_all_distractors(input_file)

    print("\n" + "=" * 50)
    print("✨ Done! All options now have similar lengths!")
