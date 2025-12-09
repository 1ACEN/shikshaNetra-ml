import librosa
import numpy as np
import os
from config.settings import SAMPLE_RATE, SPEECH_THRESHOLD_DB

class AudioAnalyzer:
    def __init__(self, audio_path):
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.audio_path = audio_path
        try:
            self.y, self.sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")

    def analyze_clarity(self):
        # Detect non-silent intervals
        non_silent_intervals = librosa.effects.split(self.y, top_db=SPEECH_THRESHOLD_DB)
        
        # Calculate total duration of non-silent parts
        non_silent_duration = sum(
            (end - start) for start, end in non_silent_intervals
        ) / self.sr
        
        total_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        if total_duration == 0:
            return 0.0
            
        pause_ratio = 1 - (non_silent_duration / total_duration)
        
        # Heuristic: Lower pause ratio is better for clarity (continuous speech)
        # However, too little pause might mean rushing. 
        # For now, we penalize excessive silence.
        # Score = 100 * (1 - pause_ratio)
        # If pause_ratio is > 0.5 (50% silence), score drops below 50.
        clarity_score = max(0, 100 * (1 - pause_ratio))
        
        return round(clarity_score, 2)

    def analyze_confidence(self):
        # 1. Pitch Stability using pyin (probabilistic YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Filter out NaN values from f0 (unvoiced parts)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            pitch_stability_score = 0
        else:
            # Standard deviation of pitch. Lower is more stable (monotone), but we want confident modulation.
            # Actually, "shaky" voice (high variance in short term) might indicate nervousness, 
            # but natural speech has pitch variation.
            # Let's look at the consistency of voicing probability as a proxy for clear speech.
            # Or simply use loudness as a major factor for "confidence".
            
            # Let's use RMS energy for loudness
            rms = librosa.feature.rms(y=self.y)[0]
            avg_rms = np.mean(rms)
            
            # Normalize RMS to a 0-100 scale (heuristic)
            # Assuming typical speech RMS might be around 0.01 to 0.1
            loudness_score = min(100, (avg_rms / 0.05) * 50) # 0.05 RMS -> 50 score, 0.1 -> 100
            
            # Pitch stability: High variance might be good (expressive) or bad (unstable).
            # Let's stick to loudness and voicing probability for now.
            avg_voiced_prob = np.mean(voiced_probs)
            voicing_score = avg_voiced_prob * 100
            
            # Combine
            confidence_score = (loudness_score * 0.6) + (voicing_score * 0.4)
            
        return round(confidence_score, 2)

    def extract_features(self):
        """
        Extract MFCCs for ML models.
        """
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        # Return the mean of MFCCs across time to get a fixed-size vector
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean.tolist()

    def analyze(self):
        return {
            "clarity_score": self.analyze_clarity(),
            "confidence_score": self.analyze_confidence(),
            "features": self.extract_features()
        }
