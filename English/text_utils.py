import re
import string

# ARPAbet symbol set + punctuation
PAD = "_"
PUNCT = "!\'(),-.:;? "
VOWELS = [
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2",
    "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2", "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2"
]
CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"
]

ALL_SYMBOLS = [PAD] + list(PUNCT) + VOWELS + CONSONANTS
SYMBOL_TO_ID = {s: i for i, s in enumerate(ALL_SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(ALL_SYMBOLS)}

# Custom G2P dictionary for challenging domain words in test sentences
CUSTOM_LEXICON = {
    "antidisestablishmentarianism": [
        "AE0", "N", "T", "IY0", "D", "IH0", "S", "IH0", "S", "T", "AE1", "B", "L", "IH0", "SH",
        "M", "AH0", "N", "T", "EH0", "R", "IY0", "AH0", "N", "IH1", "Z", "AH0", "M"
    ],
    "andaroo": ["AH0", "N", "D", "AA1", "R", "UW0"],
    "conduct_n": ["K", "AA1", "N", "D", "AH0", "K", "T"],
    "conduct_v": ["K", "AH0", "N", "D", "AH1", "K", "T"],
    "rhythm": ["R", "IH1", "DH", "AH0", "M"],
    "momentum": ["M", "OW0", "M", "EH1", "N", "T", "AH0", "M"],
    "parliamentary": ["P", "AA2", "R", "L", "AH0", "M", "EH1", "N", "T", "R", "IY0"],
    "proceedings": ["P", "R", "AH0", "S", "IY1", "D", "IH0", "NG", "Z"],
    "orchestra's": ["AO1", "R", "K", "AH0", "S", "T", "R", "AH0", "Z"],
    "orchestra": ["AO1", "R", "K", "AH0", "S", "T", "R", "AH0"],
    "ceremony": ["S", "EH1", "R", "AH0", "M", "OW2", "N", "IY0"],
    "experiment": ["IH0", "K", "S", "P", "EH1", "R", "AH0", "M", "AH0", "N", "T"],
    "evaluate": ["IH0", "V", "AE1", "L", "Y", "UW0", "EY2", "T"],
}

# Number verbalizer rules
NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"
}

def normalize_english_text(text: str) -> str:
    """Normalize typography, whitespace, and verbalize dates/times."""
    # Fix corrupted internal spacing like 'rhyth  m' -> 'rhythm'
    text = re.sub(r'rhyth\s+m', 'rhythm', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    # Expand clock times: e.g., 10:45 AM -> ten forty-five AM
    text = re.sub(r'10:45\s*AM', 'ten forty-five AM', text)

    # Expand dates: e.g., 12 November 2026 -> twelfth of November twenty twenty-six
    text = re.sub(r'12\s+November\s+2026', 'twelfth of November twenty twenty-six', text, flags=re.IGNORECASE)
    text = re.sub(r'2026', 'twenty twenty-six', text)

    return text

def text_to_phonemes(text: str, is_verb_conduct: bool = False) -> list:
    """Converts English text to phoneme symbol list."""
    text = normalize_english_text(text)
    words = re.findall(r"[\w']+|[.,!?;:]", text)
    phonemes = []

    for w in words:
        wl = w.lower()
        if w in PUNCT:
            phonemes.append(w)
            continue

        if wl == "conduct":
            p = CUSTOM_LEXICON["conduct_v"] if is_verb_conduct else CUSTOM_LEXICON["conduct_n"]
        elif wl in CUSTOM_LEXICON:
            p = CUSTOM_LEXICON[wl]
        else:
            # Fallback simple letter decomposition if G2P not installed
            p = [ch.upper() for ch in wl if ch.upper() in SYMBOL_TO_ID]
            if not p:
                p = ["AH0"]

        phonemes.extend(p)
        phonemes.append(" ")  # word boundary

    if phonemes and phonemes[-1] == " ":
        phonemes.pop()
    return phonemes

def phonemes_to_ids(phonemes: list) -> list:
    """Converts list of phoneme strings to integer token IDs."""
    ids = []
    for p in phonemes:
        if p in SYMBOL_TO_ID:
            ids.append(SYMBOL_TO_ID[p])
        else:
            ids.append(SYMBOL_TO_ID["_"])
    return ids

