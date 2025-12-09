import google.generativeai as genai
import os
import json
from config.settings import LLM_MODEL_NAME
from dotenv import load_dotenv

load_dotenv()

class ShikshaCoach:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            try:
                self.model = genai.GenerativeModel(LLM_MODEL_NAME)
            except Exception as e:
                print(f"WARNING: Failed to initialize GenerativeModel: {e}")
                self.model = None

    def _get_system_instruction(self):
        return (
            "You are an expert Pedagogical Coach and Technical Auditor for Shiksha Netra. "
            "Your goal is to evaluate a teacher based on their session transcript and computed AI scores."
        )

    def generate_comprehensive_report(self, transcript, scores_dict, topic, language="English"):

        if not self.model:
            return {"error": "GenAI model not initialized."}

        prompt = f"""
        {self._get_system_instruction()}
        
        **Session Details:**
        - Topic: {topic}
        - Language: {language}
        - AI Analysis Scores: {json.dumps(scores_dict)}
        
        **Transcript:**
        "{transcript}"
        
        **Task:**
        Analyze the session and generate a JSON report containing the following 8 distinct features:
        
        1. **performance_summary**: A 2-3 sentence executive summary of the session.
        2. **teaching_style**: Classify the style (e.g., 'Authoritative', 'Facilitator', 'Hybrid') and provide a 1-sentence explanation.
        3. **strengths**: List 3 key strengths based on the scores and transcript.
        4. **weaknesses**: List 3 areas for improvement based on the scores and transcript.
        5. **factual_accuracy_audit**: Scan the transcript for technical errors relative to the topic '{topic}'. Return a list of corrections or "No errors found".
        6. **video_titles**: Generate 3 catchy, SEO-friendly titles for this video lesson.
        7. **hashtags**: Generate 5 relevant hashtags.
        8. **multilingual_feedback**: If the target language '{language}' is NOT English, translate the 'performance_summary' into that language. If it IS English, return null.
        
        **Output Format:**
        Return ONLY the raw JSON object. Do not use markdown formatting (no ```json).
        The JSON structure must be:
        {{
            "performance_summary": "...",
            "teaching_style": {{ "style": "...", "explanation": "..." }},
            "strengths": ["...", "...", "..."],
            "weaknesses": ["...", "...", "..."],
            "factual_accuracy_audit": ["..."],
            "content_metadata": {{
                "titles": ["...", ...],
                "hashtags": ["...", ...]
            }},
            "multilingual_feedback": "..."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            return {"error": f"GenAI generation failed: {e}"}

    def chat_with_coach(self, query, context_history):

        if not self.model:
            return "I'm sorry, I cannot answer that right now because my brain (LLM) is not connected."

        prompt = f"""
        You are the Shiksha Netra Pedagogical Coach. Use the following context to answer the user's question.
        
        **Context (Previous Analysis & Transcript):**
        {json.dumps(context_history) if isinstance(context_history, dict) else context_history}
        
        **User Question:**
        "{query}"
        
        **Answer:**
        Provide a helpful, encouraging, and specific answer based on the context. Keep it concise.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def _parse_json_response(self, response_text):
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse LLM response as JSON.",
                "raw_response": response_text
            }
