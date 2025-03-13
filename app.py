import streamlit as st
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
    st.set_page_config(
        page_title="Financial Analyst Bot",
        layout="wide",
        menu_items={
            'Get Help': 'https://www.infosys.com/investors.html',
            'Report a bug': None,
            'About': "Infosys Financial Analyst v1.0"
        }
    )

    st.title("💰 INFY Financial Analyst (2022-2024)")
    st.markdown("Ask questions about Infosys financial statements from last 2 years")

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
