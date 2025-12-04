import os
import json
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from src.processors.audio_analyzer import AudioAnalyzer
from src.processors.video_analyzer import VideoAnalyzer
from src.processors.text_analyzer import TextAnalyzer
from src.genai.coach import ShikshaCoach

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """
    Extracts audio from the video file and saves it to audio_path.
    """
    print(f"Extracting audio from {video_path}...")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        if 'video' in locals():
            video.close()
        return None

def transcribe_audio(audio_path):
    """
    Mocks audio transcription. In a real app, use Whisper or an API.
    """
    print("Transcribing audio (Mock)...")
    # Return a dummy transcript relevant to a technical topic
    return """
    Hello everyone. Today we are going to discuss the fundamentals of Machine Learning.
    We will start with supervised learning and then move to unsupervised learning.
    Does anyone know what a neural network is? It is inspired by the human brain.
    We need to understand how gradients work in backpropagation.
    Let's look at some Python code examples.
    """

def process_session(video_path, topic_name="Machine Learning"):
    """
    Runs the full analysis pipeline on a video session.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    print(f"\n--- Starting Analysis for Session: {video_path} ---")
    start_time = time.time()
    
    # 1. Extract Audio
    audio_path = extract_audio(video_path)
    if not audio_path:
        return None

    try:
        # 2. Audio Analysis
        print("Running Audio Analysis...")
        audio_analyzer = AudioAnalyzer(audio_path)
        audio_results = audio_analyzer.analyze()
        print(f"Audio Scores: Clarity={audio_results['clarity_score']}, Confidence={audio_results['confidence_score']}")

        # 3. Video Analysis
        print("Running Video Analysis...")
        video_analyzer = VideoAnalyzer(video_path)
        video_results = video_analyzer.process_video()
        print(f"Video Scores: Engagement={video_results['engagement_score']}, Gesture={video_results['gesture_index']}")

        # 4. Transcription & Text Analysis
        transcript = transcribe_audio(audio_path)
        print("Running Text Analysis...")
        text_analyzer = TextAnalyzer(transcript)
        # Define some keywords based on the topic (simplified)
        keywords = ["Machine Learning", "supervised", "unsupervised", "neural network", "gradients", "Python"]
        text_results = text_analyzer.analyze(topic_name, keywords)
        print(f"Text Scores: Technical Depth={text_results['technical_depth']}, Interaction={text_results['interaction_index']}")

        # 5. Aggregate Scores
        final_report = {
            "session_id": os.path.basename(video_path),
            "topic": topic_name,
            "transcript": transcript,
            "scores": {
                "audio": audio_results,
                "video": video_results,
                "text": text_results
            }
        }

        # 6. GenAI Coach Feedback
        print("Generating Coach Feedback...")
        coach = ShikshaCoach()
        feedback = coach.generate_comprehensive_report(transcript, final_report["scores"], topic_name)
        final_report["coach_feedback"] = feedback

        print("\n--- Analysis Completed ---")
        print(f"Total Time: {round(time.time() - start_time, 2)}s")
        
        return final_report

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        return None
    finally:
        # Cleanup temp audio
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

if __name__ == "__main__":
    # Create a dummy video for testing if it doesn't exist
    import cv2
    import numpy as np
    
    test_video = "pipeline_test_video.mp4"
    if not os.path.exists(test_video):
        print("Creating dummy video for testing...")
        width, height = 640, 480
        fps = 30
        duration = 5 # seconds
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, fps, (width, height))
        
        # Create frames with some movement to simulate gestures/video
        for i in range(int(duration * fps)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Moving circle
            x = int(320 + 100 * np.sin(i * 0.1))
            y = int(240 + 50 * np.cos(i * 0.1))
            cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
            out.write(frame)
        out.release()
        
        # Add dummy audio to the video using moviepy (since cv2 doesn't write audio easily)
        # We'll just let the pipeline fail gracefully on audio extraction or 
        # we can create a separate audio file and merge it, but for simplicity
        # let's just create a dummy wav file and use that if extraction fails?
        # Actually, extract_audio will fail if video has no audio track.
        # Let's create a dummy audio file separately and tell extract_audio to use it?
        # No, let's just create a dummy wav file and use moviepy to combine them.
        
        from moviepy.editor import AudioFileClip
        import soundfile as sf
        
        # Create dummy audio
        dummy_audio_path = "dummy_source_audio.wav"
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(dummy_audio_path, audio, sr)
        
        # Merge
        video_clip = VideoFileClip(test_video)
        audio_clip = AudioFileClip(dummy_audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile("pipeline_test_video_with_audio.mp4", verbose=False, logger=None, audio_codec='aac')
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        os.remove(test_video)
        os.remove(dummy_audio_path)
        test_video = "pipeline_test_video_with_audio.mp4"

    # Run Pipeline
    report = process_session(test_video)
    
    if report:
        print("\nFinal Report JSON:")
        print(json.dumps(report, indent=2))
        
    # Cleanup test video
    if os.path.exists(test_video):
        os.remove(test_video)
