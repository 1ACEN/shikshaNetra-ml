import nltk
from sentence_transformers import SentenceTransformer, util
import re

class TextAnalyzer:
    def __init__(self, transcript):
        """
        Initialize the TextAnalyzer with a text transcript.
        Loads the sentence-transformer model.
        """
        self.transcript = transcript
        
        # Load model (this might take a moment on first run)
        # Using a lightweight model for efficiency
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformer model: {e}")
            
        # Ensure NLTK data is available (basic tokenizers)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

    def analyze_technical_depth(self, topic):
        """
        Calculate technical depth by comparing transcript similarity to the topic.
        Returns a score (0-100).
        """
        if not self.transcript or not topic:
            return 0.0
            
        # Encode transcript and topic
        embeddings = self.model.encode([self.transcript, topic])
        
        # Compute cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        
        # Convert tensor to float and scale to 0-100
        score = float(similarity[0][0]) * 100
        
        # Ensure score is non-negative
        return max(0.0, round(score, 2))

    def analyze_interaction(self):
        """
        Calculate Interaction Index based on questions and inclusive pronouns.
        Returns a score/index.
        """
        if not self.transcript:
            return 0.0
            
        # Tokenize words
        words = nltk.word_tokenize(self.transcript.lower())
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        # Count question marks (in original text or tokens)
        question_count = self.transcript.count('?')
        
        # Count inclusive pronouns
        inclusive_pronouns = {'we', 'us', 'our', "let's", 'lets'}
        pronoun_count = sum(1 for word in words if word in inclusive_pronouns)
        
        # Calculate index
        # Heuristic: 1 question or pronoun per 20 words is high interaction (5%)
        # Let's say 5% density = 100 score
        density = (question_count + pronoun_count) / total_words
        interaction_score = min(100, (density / 0.05) * 100)
        
        return round(interaction_score, 2)

    def check_topic_relevance(self, keywords):
        """
        Check if keywords are present in the transcript.
        Returns a dictionary with match counts and a relevance score.
        """
        if not self.transcript or not keywords:
            return {"matches": {}, "relevance_score": 0.0}
            
        transcript_lower = self.transcript.lower()
        matches = {}
        total_matches = 0
        
        for keyword in keywords:
            count = transcript_lower.count(keyword.lower())
            if count > 0:
                matches[keyword] = count
                total_matches += count
                
        # Relevance score: simple presence ratio or density
        # Let's use percentage of keywords found
        if len(keywords) == 0:
            relevance_score = 0.0
        else:
            relevance_score = (len(matches) / len(keywords)) * 100
            
        return {
            "matches": matches,
            "relevance_score": round(relevance_score, 2)
        }

    def analyze(self, topic, keywords):
        """
        Perform full analysis.
        """
        return {
            "technical_depth": self.analyze_technical_depth(topic),
            "interaction_index": self.analyze_interaction(),
            "topic_relevance": self.check_topic_relevance(keywords)
        }
