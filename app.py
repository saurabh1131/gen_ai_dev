import streamlit as st

import sys
sys.path.append('/mount/src/gen_ai_dev')

from utils import generate_answer

# Streamlit configuration
st.set_page_config(
    page_title="Financial Analyst Bot",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.infosys.com/investors.html',
        'About': "Infosys Financial Analyst v1.0"
    }
)

def main():
    st.title("💰 INFY Financial Analyst (2022-2024)")
    st.markdown("Ask questions about Infosys financial statements from the last 2 years.")
    
    query = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if query:
            with st.spinner("Analyzing financial reports..."):
                response, confidence = generate_answer(query)
                
            st.subheader("Answer:")
            st.markdown(f"```{response}```")
            
            st.subheader("Confidence Score:")
            st.progress(confidence)
            st.markdown(f"{confidence * 100:.1f}% relevance confidence")

if __name__ == "__main__":
    main()
