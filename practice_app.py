"""
TestGorilla AI Engineer Practice Application
Interactive quiz application built with Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from database_manager import QuestionDatabase
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TestGorilla AI Engineer Practice",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def load_database():
    return QuestionDatabase()

db = load_database()

# Initialize session state
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = False
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

def start_quiz(category, num_questions=None):
    """Start a new quiz session"""
    questions = db.get_questions_by_category(category)
    if num_questions and num_questions < len(questions):
        questions = random.sample(questions, num_questions)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.user_answers = {}
    st.session_state.show_explanation = False
    st.session_state.quiz_active = True
    st.session_state.selected_category = category
    st.session_state.start_time = datetime.now()

def submit_answer(answer_idx):
    """Submit an answer for the current question"""
    current_idx = st.session_state.current_question_idx
    question = st.session_state.quiz_questions[current_idx]

    is_correct = answer_idx == question['correct_answer']
    st.session_state.user_answers[current_idx] = {
        'answer': answer_idx,
        'correct': is_correct
    }

    # Update database progress
    db.update_progress(
        st.session_state.selected_category,
        current_idx,
        is_correct
    )

    st.session_state.show_explanation = True

def next_question():
    """Move to next question"""
    st.session_state.current_question_idx += 1
    st.session_state.show_explanation = False

def previous_question():
    """Move to previous question"""
    if st.session_state.current_question_idx > 0:
        st.session_state.current_question_idx -= 1
        st.session_state.show_explanation = False

def end_quiz():
    """End the current quiz session"""
    st.session_state.quiz_active = False

# Sidebar
with st.sidebar:
    st.title("🎯 TestGorilla Prep")
    st.markdown("---")

    # Category selection
    st.subheader("Select Category")
    categories = db.get_all_categories()

    # Group categories
    core_ai = ["Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP", "Generative AI"]
    frameworks = ["TensorFlow", "PyTorch", "Scikit-learn"]
    data_libs = ["Pandas", "NumPy"]
    programming = ["OOP", "Algorithms", "REST APIs"]
    cognitive = ["Problem Solving", "Critical Thinking"]

    # Senior-level categories (NEW)
    senior_data_ops = ["Senior NumPy - Advanced Optimization", "Senior Pandas - Production Optimization"]
    senior_frameworks = ["Senior PyTorch - Distributed Training", "Senior TensorFlow - Advanced Features"]
    senior_architecture = ["Senior Transformers - Architecture & Optimization", "Senior Deep Learning - Advanced Concepts"]
    senior_optimization = ["Senior Fine-Tuning - PEFT & LoRA", "Senior Quantization - Production Optimization"]
    senior_nlp = ["Senior Tokenization - NLP Fundamentals", "Senior Alignment - RLHF & DPO"]
    senior_engineering = ["Senior OOP Patterns - Design for ML", "Senior Algorithms - Complexity & Memory"]
    senior_system_design = ["Senior APIs - System Design for ML", "Senior Docker - Containerization for ML"]
    senior_devops = ["Senior Kubernetes - ML Workloads", "Senior Security - ML Security Best Practices"]

    # Display Senior-level categories first
    st.markdown("### 🎓 Senior AI Engineer (320 Questions)")
    st.markdown("**Data Operations**")
    for cat in senior_data_ops:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**ML Frameworks**")
    for cat in senior_frameworks:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Architecture & Optimization**")
    for cat in senior_architecture + senior_optimization:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**NLP & Alignment**")
    for cat in senior_nlp:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Software Engineering**")
    for cat in senior_engineering:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**System Design & DevOps**")
    for cat in senior_system_design + senior_devops:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("---")
    st.markdown("### 📚 Foundation Level")

    st.markdown("**Core AI/ML**")
    for cat in core_ai:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Frameworks**")
    for cat in frameworks:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Data Libraries**")
    for cat in data_libs:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Programming**")
    for cat in programming:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("**Cognitive Skills**")
    for cat in cognitive:
        if cat in categories:
            count = db.get_question_count(cat)
            if count > 0:
                if st.button(f"{cat} ({count})", key=f"cat_{cat}", use_container_width=True):
                    start_quiz(cat)

    st.markdown("---")

    # Statistics
    st.subheader("📊 Your Statistics")
    stats = db.get_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Attempted", stats['total_attempted'])
        st.metric("Correct", stats['total_correct'])
    with col2:
        st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
        st.metric("Bookmarked", stats['bookmarked_count'])

    if st.button("🔄 Reset Progress", use_container_width=True):
        # Reset user progress
        db.questions["user_progress"] = {
            "total_attempted": 0,
            "total_correct": 0,
            "category_stats": {},
            "bookmarked_questions": [],
            "attempt_history": []
        }
        db.save_database()
        st.rerun()

# Main content area
if not st.session_state.quiz_active:
    # Welcome screen
    st.title("🤖 TestGorilla AI Engineer Assessment Preparation")

    st.markdown("""
    ### Welcome to Your AI Engineer Prep Platform!

    This comprehensive practice system helps you prepare for TestGorilla AI Engineer assessments with:

    - **252 High-Quality Questions** across **15 Categories**
    - **TestGorilla-Level Difficulty**: Scenario-based questions with detailed explanations
    - **Progress Tracking**: Monitor your performance across categories
    - **Instant Feedback**: Learn from detailed explanations

    #### 📚 Available Categories
    """)

    # Display categories with stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧠 Core AI/ML**")
        for cat in core_ai:
            count = db.get_question_count(cat)
            if count > 0:
                st.markdown(f"- {cat}: **{count}**")

        st.markdown("\n**🔧 Frameworks**")
        for cat in frameworks:
            count = db.get_question_count(cat)
            if count > 0:
                st.markdown(f"- {cat}: **{count}**")

    with col2:
        st.markdown("**📊 Data Libraries**")
        for cat in data_libs:
            count = db.get_question_count(cat)
            if count > 0:
                st.markdown(f"- {cat}: **{count}**")

        st.markdown("\n**💻 Programming**")
        for cat in programming:
            count = db.get_question_count(cat)
            if count > 0:
                st.markdown(f"- {cat}: **{count}**")

    with col3:
        st.markdown("**🧩 Cognitive Skills**")
        for cat in cognitive:
            count = db.get_question_count(cat)
            if count > 0:
                st.markdown(f"- {cat}: **{count}**")

        st.markdown("\n**📈 Your Stats**")
        st.markdown(f"- Total: **{stats['total_questions']}**")
        st.markdown(f"- Attempted: **{stats['total_attempted']}**")
        st.markdown(f"- Accuracy: **{stats['accuracy']:.1f}%**")

    st.markdown("---")
    st.info("👈 **Select a category from the sidebar to start practicing!**")

    # Show category-wise performance if there's data
    if stats['category_stats']:
        st.subheader("📈 Performance by Category")

        cat_data = []
        for cat, cat_stats in stats['category_stats'].items():
            if cat_stats['attempted'] > 0:
                cat_data.append({
                    'Category': cat,
                    'Attempted': cat_stats['attempted'],
                    'Correct': cat_stats['correct'],
                    'Accuracy': (cat_stats['correct'] / cat_stats['attempted'] * 100)
                })

        if cat_data:
            df = pd.DataFrame(cat_data)
            fig = px.bar(df, x='Category', y='Accuracy',
                        title='Accuracy by Category',
                        labels={'Accuracy': 'Accuracy (%)'},
                        color='Accuracy',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

else:
    # Quiz interface
    questions = st.session_state.quiz_questions
    current_idx = st.session_state.current_question_idx

    if current_idx < len(questions):
        question = questions[current_idx]

        # Header with progress
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title(f"📝 {st.session_state.selected_category}")
        with col2:
            st.metric("Question", f"{current_idx + 1}/{len(questions)}")
        with col3:
            st.metric("Difficulty", question['difficulty'])

        # Progress bar
        progress = (current_idx + 1) / len(questions)
        st.progress(progress)

        st.markdown("---")

        # Question
        st.markdown(f"### Question {current_idx + 1}")
        st.markdown(f"**{question['question']}**")

        st.markdown("")

        # Answer options
        user_answer = st.session_state.user_answers.get(current_idx, {}).get('answer', None)

        for i, option in enumerate(question['options']):
            # Color code based on answer submission
            if st.session_state.show_explanation:
                if i == question['correct_answer']:
                    st.success(f"✅ {chr(65+i)}. {option}")
                elif user_answer == i and i != question['correct_answer']:
                    st.error(f"❌ {chr(65+i)}. {option}")
                else:
                    st.info(f"{chr(65+i)}. {option}")
            else:
                if st.button(f"{chr(65+i)}. {option}", key=f"opt_{i}",
                           disabled=st.session_state.show_explanation,
                           use_container_width=True):
                    submit_answer(i)
                    st.rerun()

        # Show explanation after submission
        if st.session_state.show_explanation:
            st.markdown("---")

            # Result
            is_correct = st.session_state.user_answers[current_idx]['correct']
            if is_correct:
                st.success("🎉 Correct!")
            else:
                st.error("❌ Incorrect")

            # Explanation
            st.markdown("### 📖 Explanation")
            st.info(question['explanation'])

            # Time estimate
            st.caption(f"⏱️ Estimated time: ~{question['time_estimate']} seconds")

        st.markdown("---")

        # Navigation buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            if current_idx > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    previous_question()
                    st.rerun()

        with col2:
            if st.session_state.show_explanation and current_idx < len(questions) - 1:
                if st.button("Next ➡️", use_container_width=True):
                    next_question()
                    st.rerun()

        with col3:
            if st.button("🔖 Bookmark", use_container_width=True):
                db.bookmark_question(st.session_state.selected_category, current_idx)
                st.toast("Question bookmarked!")

        with col4:
            if st.button("🏁 End Quiz", use_container_width=True):
                end_quiz()
                st.rerun()

    else:
        # Quiz complete
        st.title("🎊 Quiz Complete!")

        correct_answers = sum(1 for ans in st.session_state.user_answers.values() if ans['correct'])
        total_questions = len(st.session_state.user_answers)
        score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Correct Answers", correct_answers)
        with col3:
            st.metric("Score", f"{score_percentage:.1f}%")

        # Score visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Your Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Feedback
        if score_percentage >= 80:
            st.success("🌟 Excellent work! You're well-prepared for this category!")
        elif score_percentage >= 60:
            st.info("👍 Good job! Review the explanations and try again to improve.")
        else:
            st.warning("📚 Keep practicing! Review the concepts and try again.")

        st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Same Category", use_container_width=True):
                start_quiz(st.session_state.selected_category)
                st.rerun()
        with col2:
            if st.button("🏠 Back to Home", use_container_width=True):
                end_quiz()
                st.rerun()

# Footer
st.markdown("---")
st.caption("🚀 TestGorilla AI Engineer Preparation System | Built for Success")
