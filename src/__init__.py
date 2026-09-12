"""AI-Based Early Detection and Classification of Foot and Nail Conditions.

MSc dissertation project: four-class classification of foot and nail images
using transfer learning, aimed at screening support in rural healthcare
settings where a specialist is not available.

Modules
-------
config         Every experimental setting, in one place.
download_data  Fetch the public source datasets into data/raw/.
inspect_data   Audit data/raw/: counts, formats, sizes, class balance.
preprocessing  Clean, resize, split and augment into data/processed/.
train          Fine-tune MobileNetV2 (primary) and ResNet50 (comparison).
evaluate       Accuracy / precision / recall / F1 / confusion matrix.
prototype      Streamlit image-upload prediction interface.
"""

__version__ = "0.1.0"
