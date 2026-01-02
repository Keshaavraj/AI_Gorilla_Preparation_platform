"""
Master Loader for Senior AI Engineer Interview Questions - BATCH 1 (Files 1-8)
Loads 160 questions from completed checkpoints 1-4
Run this to populate the database with all senior-level questions created so far.
"""

from database_manager import QuestionDatabase

# Import all batch modules
from senior_batch1_numpy import populate_senior_numpy
from senior_batch2_pandas import populate_senior_pandas
from senior_batch3_pytorch import populate_senior_pytorch
from senior_batch4_tensorflow import populate_senior_tensorflow
from senior_batch5_transformers import populate_senior_transformers
from senior_batch6_deep_learning import populate_senior_deep_learning
from senior_batch7_finetuning import populate_senior_finetuning
from senior_batch8_quantization import populate_senior_quantization

def load_all_senior_questions_batch1():
    """Load all 160 senior questions from Batch 1 (Checkpoints 1-4)"""
    db = QuestionDatabase()

    # Create categories if they don't exist
    senior_categories = [
        "Senior NumPy - Advanced Optimization",
        "Senior Pandas - Production Optimization",
        "Senior PyTorch - Advanced Training",
        "Senior TensorFlow - Production ML",
        "Senior Transformers - Attention Mechanisms",
        "Senior Deep Learning - Advanced Architecture",
        "Senior Fine-Tuning - LoRA & PEFT Methods",
        "Senior Quantization - GPTQ, AWQ, INT8/INT4"
    ]

    for category in senior_categories:
        if category not in db.questions["categories"]:
            db.questions["categories"][category] = []

    print("=" * 80)
    print("LOADING SENIOR AI ENGINEER INTERVIEW QUESTIONS - BATCH 1")
    print("=" * 80)

    # Checkpoint 1: Data Ops (40 questions)
    print("\n📊 CHECKPOINT 1: DATA OPS")
    print("-" * 80)

    print("Loading NumPy questions...")
    numpy_questions = populate_senior_numpy()
    db.add_bulk_questions("Senior NumPy - Advanced Optimization", numpy_questions)
    print(f"✓ Added {len(numpy_questions)} NumPy questions")

    print("Loading Pandas questions...")
    pandas_questions = populate_senior_pandas()
    db.add_bulk_questions("Senior Pandas - Production Optimization", pandas_questions)
    print(f"✓ Added {len(pandas_questions)} Pandas questions")

    # Checkpoint 2: Frameworks (40 questions)
    print("\n🔥 CHECKPOINT 2: FRAMEWORKS")
    print("-" * 80)

    print("Loading PyTorch questions...")
    pytorch_questions = populate_senior_pytorch()
    db.add_bulk_questions("Senior PyTorch - Advanced Training", pytorch_questions)
    print(f"✓ Added {len(pytorch_questions)} PyTorch questions")

    print("Loading TensorFlow questions...")
    tensorflow_questions = populate_senior_tensorflow()
    db.add_bulk_questions("Senior TensorFlow - Production ML", tensorflow_questions)
    print(f"✓ Added {len(tensorflow_questions)} TensorFlow questions")

    # Checkpoint 3: Architecture (40 questions)
    print("\n🏗️  CHECKPOINT 3: ARCHITECTURE")
    print("-" * 80)

    print("Loading Transformers questions...")
    transformers_questions = populate_senior_transformers()
    db.add_bulk_questions("Senior Transformers - Attention Mechanisms", transformers_questions)
    print(f"✓ Added {len(transformers_questions)} Transformers questions")

    print("Loading Deep Learning questions...")
    dl_questions = populate_senior_deep_learning()
    db.add_bulk_questions("Senior Deep Learning - Advanced Architecture", dl_questions)
    print(f"✓ Added {len(dl_questions)} Deep Learning questions")

    # Checkpoint 4: Model Optimization (40 questions)
    print("\n⚡ CHECKPOINT 4: MODEL OPTIMIZATION")
    print("-" * 80)

    print("Loading Fine-Tuning questions...")
    finetuning_questions = populate_senior_finetuning()
    db.add_bulk_questions("Senior Fine-Tuning - LoRA & PEFT Methods", finetuning_questions)
    print(f"✓ Added {len(finetuning_questions)} Fine-Tuning questions")

    print("Loading Quantization questions...")
    quantization_questions = populate_senior_quantization()
    db.add_bulk_questions("Senior Quantization - GPTQ, AWQ, INT8/INT4", quantization_questions)
    print(f"✓ Added {len(quantization_questions)} Quantization questions")

    # Summary
    total_questions = (len(numpy_questions) + len(pandas_questions) +
                      len(pytorch_questions) + len(tensorflow_questions) +
                      len(transformers_questions) + len(dl_questions) +
                      len(finetuning_questions) + len(quantization_questions))

    print("\n" + "=" * 80)
    print(f"✅ BATCH 1 COMPLETE: Successfully loaded {total_questions} senior-level questions!")
    print("=" * 80)

    print("\n📋 CATEGORIES LOADED:")
    print("  1. Senior NumPy - Advanced Optimization (20 questions)")
    print("  2. Senior Pandas - Production Optimization (20 questions)")
    print("  3. Senior PyTorch - Advanced Training (20 questions)")
    print("  4. Senior TensorFlow - Production ML (20 questions)")
    print("  5. Senior Transformers - Attention Mechanisms (20 questions)")
    print("  6. Senior Deep Learning - Advanced Architecture (20 questions)")
    print("  7. Senior Fine-Tuning - LoRA & PEFT Methods (20 questions)")
    print("  8. Senior Quantization - GPTQ, AWQ, INT8/INT4 (20 questions)")

    print("\n🎯 COVERAGE:")
    print("  ✓ Data Operations (NumPy & Pandas)")
    print("  ✓ Deep Learning Frameworks (PyTorch & TensorFlow)")
    print("  ✓ Model Architectures (Transformers & Deep Learning)")
    print("  ✓ Model Optimization (Fine-Tuning & Quantization)")

    print("\n📝 NEXT STEPS:")
    print("  Run: python practice_app.py")
    print("  Practice these 160 questions to master:")
    print("    - Data manipulation & optimization")
    print("    - Framework-specific advanced features")
    print("    - Transformer architecture & attention mechanisms")
    print("    - Model compression & efficient training")

    return total_questions

if __name__ == "__main__":
    total = load_all_senior_questions_batch1()
    print(f"\n🚀 Ready to practice! Total questions available: {total}")
    print("=" * 80)
