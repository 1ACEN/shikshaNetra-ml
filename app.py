import streamlit as st
import os
import tempfile
import json
from src.pipeline import process_session

st.set_page_config(page_title="Shiksha Netra", page_icon="🎓", layout="wide")

st.title("🎓 Shiksha Netra - AI Pedagogical Coach")
st.markdown("Upload a teaching session video to get comprehensive AI feedback.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Session Topic", value="General")
    
    st.info("Ensure you have set your GEMINI_API_KEY in your environment variables.")

# File Uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    # Display video
    st.video(uploaded_file)
    
    if st.button("Analyze Session"):
        with st.spinner("Analyzing audio, video, and text... This may take a while."):
            # Save uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()
            
            try:
                # Run the pipeline
                report = process_session(video_path, topic_name=topic)
                
                if report:
                    st.success("Analysis Complete!")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Detailed Scores", "Coach Feedback", "Raw Data"])
                    
                    with tab1:
                        if "coach_feedback" in report and "performance_summary" in report["coach_feedback"]:
                            st.subheader("Performance Summary")
                            st.write(report["coach_feedback"]["performance_summary"])
                            
                            st.subheader("Teaching Style")
                            style = report["coach_feedback"].get("teaching_style", {})
                            if isinstance(style, dict):
                                st.info(f"**{style.get('style', 'Unknown')}**: {style.get('explanation', '')}")
                            else:
                                st.info(str(style))
                        else:
                            st.warning("Coach feedback not available (Check API Key).")
                            
                    with tab2:
                        scores = report.get("scores", {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Audio Clarity", scores.get("audio", {}).get("clarity_score", 0))
                            st.metric("Audio Confidence", scores.get("audio", {}).get("confidence_score", 0))
                            
                        with col2:
                            st.metric("Video Engagement", scores.get("video", {}).get("engagement_score", 0))
                            st.metric("Gesture Index", scores.get("video", {}).get("gesture_index", 0))
                            
                        with col3:
                            st.metric("Technical Depth", scores.get("text", {}).get("technical_depth", 0))
                            st.metric("Interaction Index", scores.get("text", {}).get("interaction_index", 0))

                    with tab3:
                        if "coach_feedback" in report:
                            fb = report["coach_feedback"]
                            
                            col_s, col_w = st.columns(2)
                            with col_s:
                                st.subheader("✅ Strengths")
                                for s in fb.get("strengths", []):
                                    st.write(f"- {s}")
                                    
                            with col_w:
                                st.subheader("⚠️ Areas for Improvement")
                                for w in fb.get("weaknesses", []):
                                    st.write(f"- {w}")
                                    
                            st.subheader("Titles & Hashtags")
                            meta = fb.get("content_metadata", {})
                            st.write("**Titles:**")
                            for t in meta.get("titles", []):
                                st.write(f"- {t}")
                            st.write("**Hashtags:** " + " ".join(meta.get("hashtags", [])))
                            
                    with tab4:
                        st.json(report)
                        
                else:
                    st.error("Analysis failed. Please check the logs.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Cleanup
                if os.path.exists(video_path):
                    os.remove(video_path)
