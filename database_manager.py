"""
Database Manager for TestGorilla AI Engineer Practice Questions
Manages questions storage, retrieval, and progress tracking
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class QuestionDatabase:
    def __init__(self, db_file='questions_db.json',
                 topics_file='topic_resources.json',
                 must_know_file='must_know_topics.json'):
        self.db_file = db_file
        self.topics_file = topics_file
        self.must_know_file = must_know_file
        self.questions = self._load_database()
        self.topics = self._load_topics()
        self.must_know = self._load_must_know()

    def _load_database(self) -> Dict:
        """Load questions from JSON file"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._initialize_database()

    def _load_topics(self) -> Dict:
        """Load topic resources from JSON file"""
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "topics": {}}

    def _load_must_know(self) -> Dict:
        """Load must-know topics from JSON file"""
        if os.path.exists(self.must_know_file):
            with open(self.must_know_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "categories": {}}

    def _initialize_database(self) -> Dict:
        """Initialize empty database structure"""
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "categories": {
                "Machine Learning": [],
                "Deep Learning": [],
                "Artificial Intelligence": [],
                "NLP": [],
                "Generative AI": [],
                "TensorFlow": [],
                "PyTorch": [],
                "Scikit-learn": [],
                "Pandas": [],
                "NumPy": [],
                "OOP": [],
                "Algorithms": [],
                "REST APIs": [],
                "Problem Solving": [],
                "Critical Thinking": []
            },
            "user_progress": {
                "total_attempted": 0,
                "total_correct": 0,
                "category_stats": {},
                "bookmarked_questions": [],
                "attempt_history": []
            }
        }

    def save_database(self):
        """Save questions to JSON file"""
        self.questions["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, indent=2, ensure_ascii=False)

    def add_question(self, category: str, question: Dict):
        """Add a single question to a category"""
        if category in self.questions["categories"]:
            self.questions["categories"][category].append(question)
            self.save_database()
        else:
            raise ValueError(f"Category '{category}' does not exist")

    def add_bulk_questions(self, category: str, questions: List[Dict]):
        """Add multiple questions to a category"""
        if category in self.questions["categories"]:
            self.questions["categories"][category].extend(questions)
            self.save_database()
        else:
            raise ValueError(f"Category '{category}' does not exist")

    def get_questions_by_category(self, category: str) -> List[Dict]:
        """Get all questions from a specific category"""
        return self.questions["categories"].get(category, [])

    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(self.questions["categories"].keys())

    def get_question_count(self, category: Optional[str] = None) -> int:
        """Get count of questions (total or by category)"""
        if category:
            return len(self.questions["categories"].get(category, []))
        return sum(len(q) for q in self.questions["categories"].values())

    def update_progress(self, category: str, question_id: int, is_correct: bool):
        """Update user progress for a question attempt"""
        self.questions["user_progress"]["total_attempted"] += 1
        if is_correct:
            self.questions["user_progress"]["total_correct"] += 1

        # Update category stats
        if category not in self.questions["user_progress"]["category_stats"]:
            self.questions["user_progress"]["category_stats"][category] = {
                "attempted": 0,
                "correct": 0
            }

        self.questions["user_progress"]["category_stats"][category]["attempted"] += 1
        if is_correct:
            self.questions["user_progress"]["category_stats"][category]["correct"] += 1

        # Add to attempt history
        self.questions["user_progress"]["attempt_history"].append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "question_id": question_id,
            "correct": is_correct
        })

        self.save_database()

    def bookmark_question(self, category: str, question_id: int):
        """Bookmark a question for later review"""
        bookmark = {"category": category, "question_id": question_id}
        if bookmark not in self.questions["user_progress"]["bookmarked_questions"]:
            self.questions["user_progress"]["bookmarked_questions"].append(bookmark)
            self.save_database()

    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        progress = self.questions["user_progress"]
        total = progress["total_attempted"]
        correct = progress["total_correct"]

        return {
            "total_questions": self.get_question_count(),
            "total_attempted": total,
            "total_correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "category_stats": progress["category_stats"],
            "bookmarked_count": len(progress["bookmarked_questions"])
        }

    def get_topic_resource(self, topic_name: str) -> Optional[Dict]:
        """Get learning resources for a specific topic"""
        return self.topics.get("topics", {}).get(topic_name, None)

    def get_all_topics(self) -> List[str]:
        """Get list of all available topics with resources"""
        return list(self.topics.get("topics", {}).keys())

    def get_must_know_for_category(self, category: str) -> Optional[Dict]:
        """Get must-know topics for a category"""
        return self.must_know.get("categories", {}).get(category, None)

    def search_topics(self, keyword: str) -> List[str]:
        """Search for topics containing the keyword"""
        topics = self.topics.get("topics", {})
        matching = []
        keyword_lower = keyword.lower()

        for topic_name, topic_data in topics.items():
            if (keyword_lower in topic_name.lower() or
                keyword_lower in topic_data.get("category", "").lower()):
                matching.append(topic_name)

        return matching


def create_question(question_text: str, options: List[str], correct_answer: int,
                   explanation: str, difficulty: str, time_estimate: int) -> Dict:
    """
    Helper function to create a properly formatted question

    Args:
        question_text: The question text
        options: List of 4 answer options
        correct_answer: Index of correct answer (0-3)
        explanation: Detailed explanation of the answer
        difficulty: 'Medium' or 'Hard'
        time_estimate: Estimated time in seconds

    Returns:
        Dictionary containing the formatted question
    """
    return {
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "difficulty": difficulty,
        "time_estimate": time_estimate
    }


if __name__ == "__main__":
    # Test the database
    db = QuestionDatabase()
    print(f"Database initialized with {db.get_question_count()} questions")
    print(f"Categories: {', '.join(db.get_all_categories())}")
