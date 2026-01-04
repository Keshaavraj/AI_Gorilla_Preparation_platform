"""
Topic Resources Page - Comprehensive Learning Resources
Progressive learning from layman → technical → formulas → code
"""

import streamlit as st
import sys
sys.path.append('..')
from database_manager import QuestionDatabase

# Page configuration
st.set_page_config(
    page_title="Topic Resources - Learn AI",
    page_icon="📚",
    layout="wide"
)

# Initialize database
@st.cache_resource
def load_database():
    return QuestionDatabase()

db = load_database()

# Header
st.title("📚 Topic Learning Resources")
st.markdown("""
Learn AI concepts progressively: **Layman Explanation → Technical Details → Formulas → Code Examples**

Perfect for junior AI engineers to build strong fundamentals and gradually tackle advanced topics.
""")

st.markdown("---")

# Sidebar for topic navigation
with st.sidebar:
    st.subheader("🔍 Find Topics")

    # Search
    search_query = st.text_input("Search topics", placeholder="e.g., backpropagation, gradient")

    if search_query:
        matching_topics = db.search_topics(search_query)
        if matching_topics:
            st.success(f"Found {len(matching_topics)} topics")
            selected_topic = st.selectbox("Select a topic", matching_topics)
        else:
            st.warning("No topics found")
            selected_topic = None
    else:
        # Show all available topics
        all_topics = db.get_all_topics()
        if all_topics:
            st.info(f"{len(all_topics)} topics available")
            selected_topic = st.selectbox("Browse all topics", [""] + sorted(all_topics))
        else:
            selected_topic = None
            st.warning("No topic resources loaded")

    st.markdown("---")

    # Must-know topics by category
    st.subheader("📖 Must-Know Topics")
    categories = ["Machine Learning", "Deep Learning", "NLP", "Generative AI",
                  "PyTorch", "TensorFlow", "NumPy", "Pandas", "MLOps", "System Design for ML"]

    selected_category = st.selectbox("View by category", [""] + categories)

# Main content area
if selected_topic and selected_topic != "":
    topic_data = db.get_topic_resource(selected_topic)

    if topic_data:
        # Topic header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.header(f"🎯 {selected_topic}")
        with col2:
            st.metric("Difficulty", topic_data.get('difficulty', 'N/A'))
        with col3:
            importance = topic_data.get('importance', 'N/A')
            color = "🔴" if importance == "Critical" else "🟡" if importance == "High" else "🟢"
            st.metric("Importance", f"{color} {importance}")

        # Category and prerequisites
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Category:** {topic_data.get('category', 'N/A')}")
        with col2:
            prereqs = topic_data.get('prerequisites', [])
            if prereqs:
                st.markdown(f"**Prerequisites:** {', '.join(prereqs)}")

        st.markdown("---")

        # Progressive learning tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗣️ Layman Explanation",
            "🔬 Technical Details",
            "📐 Formulas",
            "💻 Code Implementation",
            "🎯 Interview Tips"
        ])

        # Tab 1: Layman Explanation
        with tab1:
            st.subheader("Understanding the Concept Simply")
            layman = topic_data.get('layman_explanation', {})
            if layman:
                st.markdown(f"### {layman.get('title', '')}")
                st.info(layman.get('content', 'No explanation available'))

                # Visual analogy if available
                if 'analogy' in layman:
                    st.markdown("### 🎨 Analogy")
                    st.success(layman['analogy'])
            else:
                st.warning("Layman explanation not available yet")

        # Tab 2: Technical Details
        with tab2:
            st.subheader("Technical Understanding")
            technical = topic_data.get('technical_explanation', {})
            if technical:
                st.markdown(f"### {technical.get('title', '')}")
                st.markdown(technical.get('content', ''))

                # Key concepts
                key_concepts = technical.get('key_concepts', [])
                if key_concepts:
                    st.markdown("### 🔑 Key Concepts")
                    for i, concept in enumerate(key_concepts, 1):
                        st.markdown(f"{i}. {concept}")

                # Related topics
                related = topic_data.get('related_topics', [])
                if related:
                    st.markdown("### 🔗 Related Topics")
                    cols = st.columns(min(len(related), 3))
                    for i, rel_topic in enumerate(related):
                        with cols[i % 3]:
                            st.button(f"📘 {rel_topic}", key=f"rel_{i}")
            else:
                st.warning("Technical explanation not available yet")

        # Tab 3: Formulas
        with tab3:
            st.subheader("Mathematical Formulas")
            formulas = topic_data.get('formulas', [])
            if formulas:
                for i, formula in enumerate(formulas, 1):
                    with st.expander(f"Formula {i}: {formula.get('name', 'Unnamed')}", expanded=True):
                        # Formula
                        st.latex(formula.get('formula', ''))

                        # Explanation
                        st.markdown("**Explanation:**")
                        st.write(formula.get('explanation', ''))

                        # Variables
                        variables = formula.get('variables', {})
                        if variables:
                            st.markdown("**Variables:**")
                            for var, desc in variables.items():
                                st.markdown(f"- `{var}`: {desc}")

                        # Notes
                        if 'note' in formula:
                            st.info(f"💡 **Note:** {formula['note']}")
            else:
                st.warning("Formulas not available yet for this topic")

        # Tab 4: Code Implementation
        with tab4:
            st.subheader("Code Implementation Example")
            code_impl = topic_data.get('code_implementation', {})
            if code_impl:
                language = code_impl.get('language', 'python')

                # Explanation of code
                if 'explanation' in code_impl:
                    st.info(f"**What this code does:** {code_impl['explanation']}")

                # Simple example
                if 'simple_example' in code_impl:
                    st.markdown("### 📝 Complete Example")
                    st.code(code_impl['simple_example'], language=language)

                # Advanced example if available
                if 'advanced_example' in code_impl:
                    with st.expander("🚀 Advanced Example"):
                        st.code(code_impl['advanced_example'], language=language)

                # Try it yourself section
                st.markdown("### 🧪 Try It Yourself")
                st.markdown("""
                **Exercises to practice:**
                1. Copy the code above and run it in your local environment
                2. Modify parameters and observe changes
                3. Add print statements to understand data flow
                4. Implement a variation using different datasets
                """)
            else:
                st.warning("Code implementation not available yet")

        # Tab 5: Interview Tips
        with tab5:
            st.subheader("Interview Preparation")

            # Interview tips
            tips = topic_data.get('interview_tips', [])
            if tips:
                st.markdown("### 💡 Key Points to Mention")
                for i, tip in enumerate(tips, 1):
                    st.markdown(f"{i}. {tip}")

            # Common pitfalls
            pitfalls = topic_data.get('common_pitfalls', [])
            if pitfalls:
                st.markdown("### ⚠️ Common Pitfalls to Avoid")
                for i, pitfall in enumerate(pitfalls, 1):
                    st.error(f"{i}. {pitfall}")

            # Variants/Extensions
            variants = topic_data.get('variants', {})
            if variants:
                st.markdown("### 🔄 Variants and Extensions")
                for variant_name, variant_desc in variants.items():
                    st.markdown(f"**{variant_name}:** {variant_desc}")

            # Further reading
            reading = topic_data.get('further_reading', [])
            if reading:
                st.markdown("### 📖 Further Reading")
                for i, resource in enumerate(reading, 1):
                    st.markdown(f"{i}. {resource}")

            if not any([tips, pitfalls, variants, reading]):
                st.warning("Interview tips not available yet")

        st.markdown("---")

        # Practice section
        st.subheader("🎯 Ready to Practice?")
        col1, col2 = st.columns(2)
        with col1:
            category = topic_data.get('category', '')
            st.markdown(f"Test your knowledge with **{category}** questions")
            if st.button("📝 Go to Practice Questions", use_container_width=True):
                st.switch_page("practice_app.py")
        with col2:
            st.markdown("Explore more related topics")
            if st.button("🔍 Browse All Topics", use_container_width=True):
                st.rerun()

elif selected_category and selected_category != "":
    # Show must-know topics for selected category
    st.header(f"📖 Must-Know Topics: {selected_category}")

    must_know = db.get_must_know_for_category(selected_category)

    if must_know:
        # Priority
        if 'priority' in must_know:
            st.info(f"**Priority Level:** {must_know['priority']}")

        # Display sections
        for section_name, topics in must_know.items():
            if section_name not in ['priority'] and isinstance(topics, list):
                st.markdown(f"### {section_name.replace('_', ' ').title()}")

                # Display as columns for better layout
                cols = st.columns(2)
                for i, topic in enumerate(topics):
                    with cols[i % 2]:
                        # Check if this topic has resources
                        has_resource = topic in db.get_all_topics()
                        if has_resource:
                            if st.button(f"✅ {topic}", key=f"topic_{i}_{topic}", use_container_width=True):
                                # Load this topic
                                st.session_state.selected_topic = topic
                                st.rerun()
                        else:
                            st.markdown(f"⬜ {topic} *(resources coming soon)*")

                st.markdown("")
    else:
        st.warning(f"Must-know topics for {selected_category} not found")

else:
    # Welcome screen
    st.subheader("🎓 Welcome to Topic Resources!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 How to Use This Resource")
        st.markdown("""
        1. **Search or Browse** topics using the sidebar
        2. **Learn Progressively:**
           - Start with layman explanation
           - Move to technical details
           - Study formulas and math
           - Practice with code examples
        3. **Prepare for Interviews** with tips and common pitfalls
        4. **Practice** with related questions

        This structured approach ensures you build **deep understanding**, not just surface knowledge!
        """)

    with col2:
        st.markdown("### 📊 Available Resources")
        all_topics = db.get_all_topics()
        st.metric("Total Topics with Resources", len(all_topics))

        if all_topics:
            st.markdown("**Sample Topics:**")
            for topic in all_topics[:5]:
                st.markdown(f"- {topic}")
            if len(all_topics) > 5:
                st.markdown(f"*...and {len(all_topics) - 5} more!*")

    st.markdown("---")

    # Quick access to must-know topics
    st.subheader("🎯 Quick Access to Essential Topics")

    quick_categories = ["Machine Learning", "Deep Learning", "Generative AI", "PyTorch"]

    cols = st.columns(len(quick_categories))
    for i, cat in enumerate(quick_categories):
        with cols[i]:
            if st.button(f"📘 {cat}", use_container_width=True):
                st.session_state.selected_category = cat
                st.rerun()

    st.markdown("---")

    # Learning path
    st.subheader("🗺️ Recommended Learning Path")

    try:
        learning_path = db.must_know.get('learning_path', {})

        if learning_path:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 🌱 Beginner")
                beginner = learning_path.get('beginner_priority', [])
                for topic in beginner:
                    st.markdown(f"- {topic}")

            with col2:
                st.markdown("### 🌿 Intermediate")
                intermediate = learning_path.get('intermediate_priority', [])
                for topic in intermediate:
                    st.markdown(f"- {topic}")

            with col3:
                st.markdown("### 🌳 Advanced")
                advanced = learning_path.get('advanced_priority', [])
                for topic in advanced:
                    st.markdown(f"- {topic}")

            # Time estimates
            if 'time_estimates' in learning_path:
                st.markdown("### ⏱️ Time Estimates")
                estimates = learning_path['time_estimates']
                for level, time in estimates.items():
                    st.info(f"**{level.replace('_', ' ').title()}:** {time}")
    except:
        st.warning("Learning path information not available")

# Footer
st.markdown("---")
st.caption("📚 Comprehensive AI Engineering Learning Resources | Build Strong Fundamentals")
