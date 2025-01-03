import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BanglishCorrector:
    def __init__(self):
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            
            genai.configure(api_key=api_key)
            
            # Configure model
            generation_config = {
                "temperature": 0.3,  # Lower temperature for more consistent corrections
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 1024,
            }
            
            self.model = genai.GenerativeModel(
                model_name='gemini-pro',
                generation_config=generation_config
            )
            
            # Expanded training examples with categories
            self.corrections = {
                # Common misspellings
                'ame': 'ami',
                'amr': 'amar',
                'tmr': 'tomar',
                'tmi': 'tumi',
                'kmn': 'kemon',
                'kno': 'keno',
                
                # Verb forms
                'korchi': 'korchi',
                'korchis': 'korchish',
                'korsen': 'korchen',
                'koros': 'korish',
                'koro': 'koro',
                'korte': 'korte',
                
                # Common phrases
                'kivabe': 'kibhabe',
                'kemne': 'kemone',
                'onek': 'onek',
                'kothay': 'kothai',
                'kothai': 'kothai',
                'khub': 'khub',
                
                # Casual/Colloquial
                'accha': 'achcha',
                'acha': 'achcha',
                'hae': 'hyan',
                'thik': 'thik',
                'hocche': 'hochche',
            }
            
            # Enhanced dictionary with pattern-based corrections
            self.correction_patterns = {
                # Sound patterns
                'vowel_patterns': {
                    'o': ['o', 'oo', 'u'],
                    'i': ['i', 'ee', 'y'],
                    'e': ['e', 'ay', 'ey'],
                    'a': ['a', 'aa', 'ah']
                },
                # Common Bengali consonant patterns
                'consonant_patterns': {
                    'ch': ['ch', 'c', 'ts', 'cch'],
                    'sh': ['sh', 's', 'ss', 'cch'],
                    'th': ['th', 't', 'tth'],
                    'kh': ['kh', 'k', 'kkh'],
                    'bh': ['bh', 'b', 'v'],
                    'gh': ['gh', 'g'],
                    'jh': ['jh', 'j', 'z'],
                    'ph': ['ph', 'f'],
                },
                # Word endings
                'endings': {
                    'e': ['a', 'o', 'ey'],
                    'o': ['u', 'oo'],
                    'i': ['ee', 'y'],
                    'che': ['ce', 'che', 'chhe', 'cche'],
                }
            }
            
            # Common word forms and their variations
            self.word_forms = {
                'verbs': {
                    'present': {
                        'kori': ['kori', 'kri', 'kari'],
                        'koro': ['koro', 'kro', 'karo'],
                        'kore': ['kore', 'kre', 'kare']
                    },
                    'past': {
                        'korlam': ['korlam', 'karlam', 'korlm'],
                        'korlo': ['korlo', 'karlo', 'krlo'],
                        'korlen': ['korlen', 'karlen', 'krlen']
                    },
                    'continuous': {
                        'korchi': ['korchi', 'korchhi', 'krchi'],
                        'korcho': ['korcho', 'korchho', 'krcho'],
                        'korche': ['korche', 'korchhe', 'krche']
                    }
                },
                'pronouns': {
                    'ami': ['ami', 'ame', 'amr', 'amar'],
                    'tumi': ['tumi', 'tmi', 'tmr', 'tomar'],
                    'se': ['se', 'she', 'shey']
                }
            }
            
            logger.info("BanglishCorrector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BanglishCorrector: {e}")
            raise

    async def correct_word(self, word: str) -> str:
        """Correct a single Banglish word using Gemini API"""
        try:
            # First check if it's in our basic corrections
            if word.lower() in self.corrections:
                return self.corrections[word.lower()]

            # Enhanced prompt with more context and examples
            prompt = f"""
            You are a Banglish (Bengali language written with English letters) text correction expert. Your task is to correct informal or incorrect Banglish spellings to their standard form.

            Context:
            - Banglish is Bengali written using English letters
            - Many words have multiple common misspellings
            - Corrections should maintain Banglish format (not Bengali script)
            - Consider both formal and informal writing styles

            Common patterns to correct:
            1. Missing 'h' in aspirated sounds: 'k' -> 'kh', 't' -> 'th'
            2. 'v' vs 'b' confusion: 'valo' -> 'bhalo'
            3. Ending variations: 'korcho' -> 'korchho'
            4. Regional variations: 'achen' -> 'achhen'

            Example corrections:
            Basic Words:
            - 'ame/ami/amr' -> 'ami'
            - 'tmi/tumi/tomr' -> 'tumi'
            - 'kmon/kmn' -> 'kemon'
            
            Verb Forms:
            - 'korta/karta' -> 'korte'
            - 'karchi/korchhi' -> 'korchi'
            - 'kora' -> 'kore'
            
            Common Phrases:
            - 'kivabe' -> 'kibhabe'
            
            Casual Forms:
            - 'hae' -> 'hya'
            - 'acha' -> 'achcha'

            Common error cases:
            - If you find a word has 'ss' in it, suggest that part of the word with 'cch'
            - If you find a word has 'sh' in it, suggest suggest that part of the word  with 'cch'
            - If you find a word has 'bh' in it, suggest suggest that part of the word  with 'v'
            - If you find a word has 'kk' in it, suggest suggest that part of the word  with 'kkh'
            - If you find a word has 'ch' in it, suggest suggest that part of the word  with 'cch'
            - If you find a word ends with 'a' in it, suggest suggest that part of the word  with 'e'

            Rules:
            1. Only correct if it feels like a misspelling
            2. Preserve dialectal variations if they're valid
            3. Return only the corrected word
            4. Keep informal/casual forms if they're commonly used
            5. Don't convert to Bengali script
            You have to correct this Banglish word: '{word}'

            
            """

            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return word
                
            corrected = response.text.strip()
            corrected = corrected.replace('"', '').replace("'", "").strip()
            
            # Additional validation
            if len(corrected.split()) > 1 or not corrected:
                return word
            
            # Prevent over-correction of valid words
            if word.lower() == corrected.lower():
                return word
                
            return corrected
            
        except Exception as e:
            logger.error(f"Error correcting word '{word}': {e}")
            return word

    async def correct_text(self, text: str) -> str:
        """Correct full Banglish text with context awareness"""
        try:
            # Split text into sentences
            sentences = text.replace('।', '.').split('.')
            corrected_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Split sentence into words
                words = sentence.strip().split()
                corrected_words = []
                
                # Process words with context
                for i, word in enumerate(words):
                    # Skip punctuation
                    if word in [',', '!', '?']:
                        corrected_words.append(word)
                        continue
                    
                    # Get surrounding words for context
                    prev_word = words[i-1] if i > 0 else None
                    next_word = words[i+1] if i < len(words)-1 else None
                    
                    # First try pattern-based correction
                    corrected = await self.analyze_word(word)
                    
                    # If word changed, verify with Gemini API
                    if corrected != word:
                        verified = await self.verify_correction(word, corrected, prev_word, next_word)
                        corrected_words.append(verified)
                    else:
                        # If no pattern match, use Gemini API
                        ai_corrected = await self.correct_word(word, prev_word, next_word)
                        corrected_words.append(ai_corrected)
                
                # Rejoin sentence
                corrected_sentence = ' '.join(corrected_words)
                corrected_sentences.append(corrected_sentence)
            
            # Rejoin text with proper punctuation
            return '. '.join(corrected_sentences).replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
            
        except Exception as e:
            logger.error(f"Error correcting text: {e}")
            return text 

    async def analyze_word(self, word: str) -> str:
        """Analyze word structure and apply appropriate corrections"""
        try:
            # Convert to lowercase for analysis
            word = word.lower()
            
            # Check if it's a known word form
            for category in self.word_forms:
                for correct_form, variations in self.word_forms[category].items():
                    if word in variations:
                        return correct_form

            # Apply pattern-based corrections
            corrected_word = word
            
            # Check consonant patterns
            for correct, variants in self.correction_patterns['consonant_patterns'].items():
                for variant in variants:
                    if variant in word:
                        corrected_word = corrected_word.replace(variant, correct)
            
            # Check vowel patterns
            for correct, variants in self.correction_patterns['vowel_patterns'].items():
                for variant in variants:
                    if variant in word:
                        # Only replace if it's not part of another pattern
                        if not any(variant in p for p in self.correction_patterns['consonant_patterns']):
                            corrected_word = corrected_word.replace(variant, correct)
            
            # Check word endings
            for correct, variants in self.correction_patterns['endings'].items():
                for variant in variants:
                    if word.endswith(variant):
                        corrected_word = corrected_word[:-len(variant)] + correct
            
            return corrected_word
        
        except Exception as e:
            logger.error(f"Error analyzing word '{word}': {e}")
            return word 

    async def verify_correction(self, original: str, corrected: str, prev_word: str = None, next_word: str = None) -> str:
        """Verify correction using Gemini API with context"""
        try:
            context = f"{prev_word + ' ' if prev_word else ''}{original}{' ' + next_word if next_word else ''}"
            
            prompt = f"""
            As a Banglish (Bengali written in English) correction expert, verify this correction:
            
            Original word: {original}
            Suggested correction: {corrected}
            Context: "{context}"
            
            Instructions:
            1. Consider the surrounding words for context
            2. Check if the correction maintains the intended meaning
            3. Verify if the correction follows Bengali language patterns
            4. Consider common dialectal variations
            
            Return ONLY ONE of these two words (no other text):
            - Return "{original}" if the original is correct or if unsure
            - Return "{corrected}" if the correction is better
            """

            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return original
                
            verified = response.text.strip().lower()
            verified = verified.replace('"', '').replace("'", "").strip()
            
            # If response is neither original nor corrected, return original
            if verified not in [original.lower(), corrected.lower()]:
                return original
                
            return verified
            
        except Exception as e:
            logger.error(f"Error verifying correction for '{original}': {e}")
            return original 