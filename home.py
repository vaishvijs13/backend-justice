import streamlit as st
from embeddings import process, search_similar

st.title("Deposition Video Search")

# upload section
uploaded_video = st.file_uploader("Upload Video:", type=['mp4', 'wav', 'mov'])

if uploaded_video:
    with st.spinner("Processing the video..."):
        temp_path = f"temp_{uploaded_video.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())
        
        process(temp_path)
        st.success("Video has been processed and indexed successfully!")

        os.remove(temp_path)

# search section
st.subheader("Search for content")
query = st.text_input("Enter your query:")
if query:
    with st.spinner("Searching for relevant clips..."):
        results = search_similar(query, top_k=5) # top 5 similar ones right now
        
        if results:
            st.write("Top matching clips:")
            for result in results:
                video_path = result['video_filename']
                text_snippet = result['text']
                st.video(video_path) # show the actual video
        else:
            st.warning("No relevant clips found for the given query.")
