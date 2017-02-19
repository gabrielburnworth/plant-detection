"""Capture and Detect commands to load as farmware."""

from Plant_Detection import Plant_Detection

if __name__ == "__main__":
    PD = Plant_Detection(from_env_var=True, verbose=False)
    PD.detect_plants()
