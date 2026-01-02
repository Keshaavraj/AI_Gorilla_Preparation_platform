"""
COMPLETE LOADER: All 320 Senior-Level AI Engineer Interview Questions
========================================================================

This script loads ALL 320 questions from 16 batch files into the question database.

Coverage:
- Batch 1: NumPy (20Q) - Advanced optimization, strides, broadcasting
- Batch 2: Pandas (20Q) - Production optimization, chunking, memory
- Batch 3: PyTorch (20Q) - DDP, autograd, memory optimization
- Batch 4: TensorFlow (20Q) - Distributed training, XLA, tf.function
- Batch 5: Transformers (20Q) - Flash Attention, KV cache, attention
- Batch 6: Deep Learning (20Q) - Normalization, activations, architectures
- Batch 7: Fine-Tuning (20Q) - LoRA, QLoRA, PEFT methods
- Batch 8: Quantization (20Q) - GPTQ, AWQ, INT4/INT8
- Batch 9: Tokenization (20Q) - BPE, WordPiece, SentencePiece
- Batch 10: Alignment (20Q) - RLHF, DPO, PPO, reward modeling
- Batch 11: OOP Patterns (20Q) - Design patterns for ML systems
- Batch 12: Algorithms (20Q) - Space complexity, memory layout
- Batch 13: APIs (20Q) - System design for ML serving
- Batch 14: Docker (20Q) - Containerization for ML/AI
- Batch 15: Kubernetes (20Q) - K8s for ML workloads
- Batch 16: Security (20Q) - ML security best practices

Total: 320 questions across 16 topics
"""

import sys
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
from senior_batch9_tokenization import populate_senior_tokenization
from senior_batch10_alignment import populate_senior_alignment
from senior_batch11_oop_patterns import populate_senior_oop_patterns
from senior_batch12_algorithms import populate_senior_algorithms
from senior_batch13_apis import populate_senior_apis
from senior_batch14_docker import populate_senior_docker
from senior_batch15_kubernetes import populate_senior_kubernetes
from senior_batch16_security import populate_senior_security


def load_all_320_questions():
    """
    Load all 320 senior-level questions into the database

    Returns:
        tuple: (total_questions_loaded, categories_created)
    """
    print("=" * 80)
    print("LOADING ALL 320 SENIOR-LEVEL AI ENGINEER INTERVIEW QUESTIONS")
    print("=" * 80)
    print()

    # Initialize database
    db = QuestionDatabase()

    # Define all categories
    categories = [
        "Senior NumPy - Advanced Optimization",
        "Senior Pandas - Production Optimization",
        "Senior PyTorch - Distributed Training",
        "Senior TensorFlow - Advanced Features",
        "Senior Transformers - Architecture & Optimization",
        "Senior Deep Learning - Advanced Concepts",
        "Senior Fine-Tuning - PEFT & LoRA",
        "Senior Quantization - Production Optimization",
        "Senior Tokenization - NLP Fundamentals",
        "Senior Alignment - RLHF & DPO",
        "Senior OOP Patterns - Design for ML",
        "Senior Algorithms - Complexity & Memory",
        "Senior APIs - System Design for ML",
        "Senior Docker - Containerization for ML",
        "Senior Kubernetes - ML Workloads",
        "Senior Security - ML Security Best Practices",
    ]

    # Create categories if they don't exist
    print("Step 1: Creating categories...")
    for category in categories:
        if category not in db.questions["categories"]:
            db.questions["categories"][category] = []
    print(f"✓ {len(categories)} categories ready\n")

    # Define all batch loaders
    batch_loaders = [
        ("Senior NumPy - Advanced Optimization", populate_senior_numpy, "NumPy"),
        ("Senior Pandas - Production Optimization", populate_senior_pandas, "Pandas"),
        ("Senior PyTorch - Distributed Training", populate_senior_pytorch, "PyTorch"),
        ("Senior TensorFlow - Advanced Features", populate_senior_tensorflow, "TensorFlow"),
        ("Senior Transformers - Architecture & Optimization", populate_senior_transformers, "Transformers"),
        ("Senior Deep Learning - Advanced Concepts", populate_senior_deep_learning, "Deep Learning"),
        ("Senior Fine-Tuning - PEFT & LoRA", populate_senior_finetuning, "Fine-Tuning"),
        ("Senior Quantization - Production Optimization", populate_senior_quantization, "Quantization"),
        ("Senior Tokenization - NLP Fundamentals", populate_senior_tokenization, "Tokenization"),
        ("Senior Alignment - RLHF & DPO", populate_senior_alignment, "Alignment"),
        ("Senior OOP Patterns - Design for ML", populate_senior_oop_patterns, "OOP Patterns"),
        ("Senior Algorithms - Complexity & Memory", populate_senior_algorithms, "Algorithms"),
        ("Senior APIs - System Design for ML", populate_senior_apis, "APIs"),
        ("Senior Docker - Containerization for ML", populate_senior_docker, "Docker"),
        ("Senior Kubernetes - ML Workloads", populate_senior_kubernetes, "Kubernetes"),
        ("Senior Security - ML Security Best Practices", populate_senior_security, "Security"),
    ]

    # Load all batches
    print("Step 2: Loading questions from all 16 batches...\n")
    total_loaded = 0

    for i, (category, loader_func, display_name) in enumerate(batch_loaders, 1):
        try:
            print(f"[{i}/16] Loading {display_name}...", end=" ")
            questions = loader_func()
            db.add_bulk_questions(category, questions)
            total_loaded += len(questions)
            print(f"✓ {len(questions)} questions loaded")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return None, None

    print()
    print("=" * 80)
    print("LOADING COMPLETE!")
    print("=" * 80)
    print(f"✅ Successfully loaded {total_loaded} senior-level questions")
    print(f"✅ Created {len(categories)} categories")
    print()

    # Summary by checkpoint
    print("Summary by Checkpoint:")
    print("-" * 80)
    print("CHECKPOINT 1-5 (Data Science & ML Frameworks): 200 questions")
    print("  ├─ NumPy (20Q) - Strides, broadcasting, vectorization")
    print("  ├─ Pandas (20Q) - Chunking, memory optimization")
    print("  ├─ PyTorch (20Q) - DDP, autograd, custom layers")
    print("  ├─ TensorFlow (20Q) - XLA, distributed strategies")
    print("  ├─ Transformers (20Q) - Flash Attention, KV cache")
    print("  ├─ Deep Learning (20Q) - Normalization, architectures")
    print("  ├─ Fine-Tuning (20Q) - LoRA, QLoRA, PEFT")
    print("  ├─ Quantization (20Q) - GPTQ, AWQ, INT4/INT8")
    print("  ├─ Tokenization (20Q) - BPE, WordPiece, multilingual")
    print("  └─ Alignment (20Q) - RLHF, DPO, PPO")
    print()
    print("CHECKPOINT 6 (Software Engineering): 40 questions")
    print("  ├─ OOP Patterns (20Q) - Factory, Strategy, Observer")
    print("  └─ Algorithms (20Q) - Space complexity, memory layout")
    print()
    print("CHECKPOINT 7 (System Design): 40 questions")
    print("  ├─ APIs (20Q) - Throughput, latency, caching")
    print("  └─ Docker (20Q) - GPU passthrough, multi-stage builds")
    print()
    print("CHECKPOINT 8 (DevOps & Security): 40 questions")
    print("  ├─ Kubernetes (20Q) - shm-size, GPU scheduling, PVCs")
    print("  └─ Security (20Q) - OWASP LLM Top 10, prompt injection")
    print()
    print("-" * 80)
    print(f"TOTAL: {total_loaded} questions across 16 topics")
    print()

    # Print next steps
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Start the practice application:")
    print("   python3 practice_app.py")
    print()
    print("2. Select a category to practice")
    print()
    print("3. Track your progress through all 320 questions")
    print()
    print("4. Review explanations for Senior-level depth")
    print()
    print("=" * 80)

    return total_loaded, len(categories)


def verify_question_counts():
    """
    Verify that each batch has exactly 20 questions

    Returns:
        bool: True if all batches have 20 questions, False otherwise
    """
    print("=" * 80)
    print("VERIFYING QUESTION COUNTS")
    print("=" * 80)
    print()

    batch_loaders = [
        (populate_senior_numpy, "NumPy"),
        (populate_senior_pandas, "Pandas"),
        (populate_senior_pytorch, "PyTorch"),
        (populate_senior_tensorflow, "TensorFlow"),
        (populate_senior_transformers, "Transformers"),
        (populate_senior_deep_learning, "Deep Learning"),
        (populate_senior_finetuning, "Fine-Tuning"),
        (populate_senior_quantization, "Quantization"),
        (populate_senior_tokenization, "Tokenization"),
        (populate_senior_alignment, "Alignment"),
        (populate_senior_oop_patterns, "OOP Patterns"),
        (populate_senior_algorithms, "Algorithms"),
        (populate_senior_apis, "APIs"),
        (populate_senior_docker, "Docker"),
        (populate_senior_kubernetes, "Kubernetes"),
        (populate_senior_security, "Security"),
    ]

    all_valid = True
    total = 0

    for i, (loader_func, display_name) in enumerate(batch_loaders, 1):
        questions = loader_func()
        count = len(questions)
        total += count

        status = "✓" if count == 20 else "✗"
        print(f"{status} Batch {i:2d} ({display_name:20s}): {count:2d} questions")

        if count != 20:
            all_valid = False

    print()
    print("-" * 80)
    print(f"Total questions: {total}")
    print(f"Expected: 320")
    print(f"Status: {'✓ PASS' if total == 320 and all_valid else '✗ FAIL'}")
    print("=" * 80)
    print()

    return all_valid and total == 320


if __name__ == "__main__":
    print()

    # Verify counts first
    if verify_question_counts():
        print()
        # Load all questions
        total, categories = load_all_320_questions()

        if total == 320:
            print("🎉 SUCCESS! All 320 questions loaded successfully!")
            print()
            print("You are now ready to practice for Senior AI Engineer interviews!")
            print()
            sys.exit(0)
        else:
            print("❌ ERROR: Expected 320 questions but loaded", total)
            sys.exit(1)
    else:
        print("❌ ERROR: Question count verification failed")
        sys.exit(1)
