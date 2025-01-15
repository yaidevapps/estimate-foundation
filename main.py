import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiEstimator

# Page configuration
st.set_page_config(
    page_title="Foundation Cost Estimator",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'images_analyzed' not in st.session_state:
    st.session_state.images_analyzed = False
if 'current_images' not in st.session_state:
    st.session_state.current_images = []

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        estimator = GeminiEstimator(api_key)
    else:
        estimator = GeminiEstimator()
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = estimator.start_chat()
        st.session_state.images_analyzed = False
        st.session_state.current_images = []
        st.rerun()

# Main title
st.title("🏗️ Foundation Cost Estimator")
st.markdown("""
Upload foundation plan images to automatically estimate concrete volumes and costs. 
Our system calculates estimates based on standard rates:
- Foundation footings and walls: $900/yard
- Retaining walls: $1,100/yard
""")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = estimator.start_chat()

# File uploader
uploaded_files = st.file_uploader("Upload foundation plan images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Main interface
if uploaded_files:
    # Create columns for image display
    cols = st.columns(3)
    
    # Display images in a grid
    for idx, uploaded_file in enumerate(uploaded_files):
        col_idx = idx % 3
        with cols[col_idx]:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Plan Image {idx + 1}", use_container_width=True)
    
    # Analyze button
    if not st.session_state.images_analyzed and len(uploaded_files) > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Generate Foundation Estimate", type="primary"):
                with st.spinner("Analyzing plans and calculating foundation costs..."):
                    # Store images for reference
                    st.session_state.current_images = [Image.open(file) for file in uploaded_files]
                    
                    # Get initial analysis
                    report = estimator.analyze_images(st.session_state.current_images, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.images_analyzed = True
                    st.rerun()

# Display chat history
st.markdown("### 💬 Foundation Estimate Analysis Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.images_analyzed:
    if prompt := st.chat_input("Ask questions about the foundation estimate..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = estimator.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use This Foundation Cost Estimator
1. Upload clear images of your foundation plans (multiple images supported)
2. Click "Generate Foundation Estimate" to get an automated calculation
3. Review the detailed breakdown of concrete volumes and costs
4. Ask questions about specific components or calculations
5. Use the "Clear Chat History" button in the sidebar to start fresh

Example questions you can ask:
- Can you break down the footing volume calculations in more detail?
- What assumptions were made about the wall heights?
- How would soil conditions impact this estimate?
- Can you explain the complexity factors used?
- What considerations were made for drainage requirements?
- How do weather conditions affect the cost?
""")

# Download button for chat history
if st.session_state.messages:
    chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Foundation Estimate Report",
        chat_history,
        file_name="foundation_estimate.txt",
        mime="text/plain"
    )