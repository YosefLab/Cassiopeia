"""
Stores constants for the ProcessingPipeline module
"""

DNA_SUBSTITUTION_MATRIX = {
                    "A": {"A": 5, "T": -4, "C": -4, "G": -4, "Z": 0, "N": 0},
                    "T": {"A": -4, "T": 5, "C": -4, "G": -4, "Z": 0, "N": 0},
                    "C": {"A": -4, "T": -4, "C": 5, "G": -4, "Z": 0, "N": 0},
                    "G": {"A": -4, "T": -4, "C": -4, "G": 5, "Z": 0, "N": 0},
                    "Z": {"A": 0, "T": 0, "C": 0, "G": 0, "Z": 0, "N": 0},
                    "N": {"A": 0, "T": 0, "C": 0, "G": 0, "Z": 0, "N": 0},
}                    