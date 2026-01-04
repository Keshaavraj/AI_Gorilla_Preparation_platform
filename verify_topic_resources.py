#!/usr/bin/env python3
"""
Verification script for Topic Resources System
Tests that all components are working correctly
"""

import sys
import json
from pathlib import Path

def test_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} MISSING: {filepath}")
        return False

def test_json_valid(filepath, description):
    """Check if JSON file is valid"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ {description} is valid JSON")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ {description} has JSON errors: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading {description}: {e}")
        return None

def test_database_manager():
    """Test database manager imports and functions"""
    try:
        from database_manager import QuestionDatabase
        print("✅ database_manager.py imports successfully")

        db = QuestionDatabase()
        print("✅ QuestionDatabase instantiates successfully")

        # Test new methods
        topics = db.get_all_topics()
        print(f"✅ get_all_topics() works: {len(topics)} topics found")

        if topics:
            first_topic = topics[0]
            resource = db.get_topic_resource(first_topic)
            if resource:
                print(f"✅ get_topic_resource() works for '{first_topic}'")
            else:
                print(f"⚠️  get_topic_resource() returned None for '{first_topic}'")

        categories = list(db.must_know.get('categories', {}).keys())
        print(f"✅ Must-know categories loaded: {len(categories)} categories")

        return True
    except ImportError as e:
        print(f"❌ Failed to import database_manager: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing database_manager: {e}")
        return False

def test_topic_structure(topic_data, topic_name):
    """Validate topic resource structure"""
    required_fields = [
        'category',
        'difficulty',
        'importance',
        'layman_explanation',
        'technical_explanation',
        'formulas',
        'code_implementation'
    ]

    missing = []
    for field in required_fields:
        if field not in topic_data:
            missing.append(field)

    if missing:
        print(f"⚠️  Topic '{topic_name}' missing fields: {', '.join(missing)}")
        return False
    else:
        print(f"✅ Topic '{topic_name}' has all required fields")
        return True

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("Topic Resources System Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Test 1: Check files exist
    print("📁 Checking required files...")
    print("-" * 60)
    files = [
        ('topic_resources.json', 'Topic resources file'),
        ('must_know_topics.json', 'Must-know topics file'),
        ('database_manager.py', 'Database manager'),
        ('practice_app.py', 'Practice app'),
        ('pages/1_📚_Topic_Resources.py', 'Topic resources page')
    ]

    for filepath, desc in files:
        if not test_file_exists(filepath, desc):
            all_passed = False
    print()

    # Test 2: Validate JSON files
    print("📋 Validating JSON files...")
    print("-" * 60)

    topic_data = test_json_valid('topic_resources.json', 'topic_resources.json')
    if topic_data:
        topics = topic_data.get('topics', {})
        print(f"   Found {len(topics)} topics: {', '.join(topics.keys())}")
    else:
        all_passed = False
    print()

    must_know_data = test_json_valid('must_know_topics.json', 'must_know_topics.json')
    if must_know_data:
        categories = must_know_data.get('categories', {})
        print(f"   Found {len(categories)} categories")
    else:
        all_passed = False
    print()

    # Test 3: Test database manager
    print("🔧 Testing database_manager.py...")
    print("-" * 60)
    if not test_database_manager():
        all_passed = False
    print()

    # Test 4: Validate topic structures
    if topic_data and 'topics' in topic_data:
        print("🔍 Validating topic structures...")
        print("-" * 60)
        for topic_name, topic_info in topic_data['topics'].items():
            if not test_topic_structure(topic_info, topic_name):
                all_passed = False
        print()

    # Test 5: Check learning paths
    if must_know_data and 'learning_path' in must_know_data:
        print("🗺️  Checking learning paths...")
        print("-" * 60)
        lp = must_know_data['learning_path']
        for level in ['beginner_priority', 'intermediate_priority', 'advanced_priority']:
            if level in lp:
                print(f"✅ {level}: {len(lp[level])} topics")
            else:
                print(f"⚠️  {level} not found")
        print()

    # Final summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("🚀 System is ready to use!")
        print()
        print("Next steps:")
        print("1. Run: streamlit run practice_app.py")
        print("2. Click '📚 Learn Topics' in the sidebar")
        print("3. Explore available topic resources")
        print("4. Add more topics using EXTENDING_TOPICS_GUIDE.md")
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Please check the errors above and fix them.")
        print("Refer to TOPIC_RESOURCES_README.md for help.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
