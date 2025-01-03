import logging
from typing import List, Optional, Dict, Tuple
from rapidfuzz import fuzz, process
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BanglishService:
    def __init__(self):
        try:
            self.mapping_file = Path("data/banglish_mapping.json")
            # Dictionary to store misspelled -> correct mappings
            self.spelling_corrections = self._load_spelling_corrections()
            # Set of correct Banglish words
            self.correct_words = set(self.spelling_corrections.values())
            logger.info("BanglishService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BanglishService: {e}")
            raise

    def _load_spelling_corrections(self) -> Dict[str, str]:
        """Load or create Banglish spelling corrections"""
        default_corrections = {
            # Common misspellings -> correct Banglish
            "ame": "ami",
            "amee": "ami",
            "tume": "tumi",
            "tumee": "tumi",
            "apne": "apni",
            "kemun": "kemon",
            "kmn": "kemon",
            "valo": "bhalo",
            "balo": "bhalo",
            "kothai": "kothay",
            "koi": "kothay",
            "ke": "ki",
            "kee": "ki",
            "krcho": "korcho",
            "krchen": "korchen",
            "bol": "bolo",
            "blun": "bolun",
            "dhk": "dhaka",
            "dkh": "dhaka",
            "bangali": "bangla",
            "ing": "english",
            "sekhi": "shikhi",
            "sikhi": "shikhi",
            "sekhbo": "shikhbo",
            "sikhbo": "shikhbo",
            "jni": "jani",
            "janena": "janina",
            "buji": "bujhi",
            "bujina": "bujhina",
            "bujhena": "bujhina",
            "kub": "khub",
            "onk": "onek",
            "sundur": "sundor",
            "sundr": "sundor",
            "shundr": "sundor",
            "shundur": "sundor",
            "habijbi": "habijabi",
            "hbijbi": "habijabi",
            "ktha": "kotha",
            "kthay": "kothay",
            "kno": "keno",
            "kenoo": "keno",
            "hbe": "hobe",
            "hby": "hobe",
            "hoby": "hobe",
            "hobey": "hobe",
            "kore": "koro",
            "kro": "koro",
            "krr": "koro",
            "krun": "korun",
            "krn": "korun"
        }

        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.mapping_file.parent.mkdir(exist_ok=True)
                with open(self.mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(default_corrections, f, ensure_ascii=False, indent=2)
                return default_corrections
        except Exception as e:
            logger.error(f"Error loading spelling corrections: {e}")
            return default_corrections

    def _find_best_match(self, word: str, threshold: int = 80) -> Tuple[Optional[str], int]:
        """Find best matching correct word using fuzzy matching"""
        try:
            # First check exact matches in spelling corrections
            if word in self.spelling_corrections:
                return self.spelling_corrections[word], 100
            
            # Then try fuzzy matching with correct words
            match = process.extractOne(
                word,
                self.correct_words,
                scorer=fuzz.WRatio,
                score_cutoff=threshold
            )
            
            if match:
                return match[0], match[1]
            return None, 0
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None, 0

    async def get_correction(self, text: str) -> Optional[str]:
        """Get spelling correction for Banglish text"""
        try:
            words = text.lower().split()
            corrected_words = []
            
            for word in words:
                best_match, score = self._find_best_match(word)
                if best_match:
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
            
            corrected_text = ' '.join(corrected_words)
            return corrected_text if corrected_text != text else None
            
        except Exception as e:
            logger.error(f"Error getting correction: {e}")
            return None

    async def get_suggestions(self, text: str) -> List[str]:
        """Get possible spelling suggestions using fuzzy matching"""
        try:
            words = text.lower().split()
            suggestions = set()
            
            for word in words:
                # Get multiple close matches for each word
                matches = process.extract(
                    word,
                    self.correct_words,
                    scorer=fuzz.WRatio,
                    limit=3,
                    score_cutoff=65  # Lower threshold for suggestions
                )
                
                if matches:
                    # Create suggestions by replacing the word
                    for match, score in matches:
                        new_words = words.copy()
                        new_words[words.index(word)] = match
                        suggestion = ' '.join(new_words)
                        suggestions.add(suggestion)
                
                # Also add exact mapping if available
                if word in self.spelling_corrections:
                    new_words = words.copy()
                    new_words[words.index(word)] = self.spelling_corrections[word]
                    suggestion = ' '.join(new_words)
                    suggestions.add(suggestion)
            
            return list(suggestions)[:5]  # Return top 5 unique suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []

    def get_bangla_for_suggestion(self, suggestion: str) -> str:
        """Get Bangla translation for a suggestion"""
        try:
            # Load Bangla translations
            bangla_map = {
                "ami": "আমি",
                "tumi": "তুমি",
                "apni": "আপনি",
                "kemon": "কেমন",
                "acho": "আছো",
                "achen": "আছেন",
                "bhalo": "ভালো",
                "kothay": "কোথায়",
                "ki": "কি",
                "korcho": "করছো",
                "korchen": "করছেন",
                "bolo": "বলো",
                "bolun": "বলুন",
                "dhaka": "ঢাকা",
                "bangla": "বাংলা",
                "english": "ইংরেজি",
                "shikhi": "শিখি",
                "shikhbo": "শিখবো",
                "jani": "জানি",
                "janina": "জানিনা",
                "bujhi": "বুঝি",
                "bujhina": "বুঝিনা",
                "khub": "খুব",
                "onek": "অনেক",
                "sundor": "সুন্দর",
                "kotha": "কথা",
                "keno": "কেন",
                "hobe": "হবে",
                "koro": "করো",
                "korun": "করুন",
                "habijabi": "হাবিজাবি"
            }
            
            words = suggestion.split()
            bangla_words = []
            
            for word in words:
                bangla_words.append(bangla_map.get(word, word))
            
            return ' '.join(bangla_words)
        except Exception as e:
            logger.error(f"Error getting Bangla translation: {e}")
            return suggestion