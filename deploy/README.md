---
title: Foot and Nail Screening
emoji: 🦶
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: MSc dissertation prototype — foot and nail condition screening
---

# Foot and nail screening

A research prototype from an MSc dissertation, *AI-Based Early Detection and
Classification of Foot and Nail Conditions Using Transfer Learning for Rural
Healthcare* (Gurubasavaraj Harlapur, Middlesex University Dubai).

Photograph a foot or a nail and the model sorts it into one of four classes —
healthy, nail fungal infection, wound or injury, ulcer — reports a calibrated
confidence, shows which part of the image drove the decision, and recommends how
soon to seek care. Below a fitted confidence threshold it declines to name a
condition rather than guess.

**This is not a medical device.** It must not be used for diagnosis or to decide
treatment.

## What is deployed here

MobileNetV2, the architecture the dissertation recommends for deployment: 98.32
percent on the held-out test set, a tenth the size of the ResNet50 it was
compared against and roughly twice as fast on CPU, with no statistically
significant difference in accuracy between them.

Confidence is temperature-scaled using the value fitted on the validation split,
because raw softmax outputs from a deep network are systematically overconfident.

## What it does not do

It was trained on clinical photographs. On photographs taken with an ordinary
phone in ordinary indoor lighting it is measurably less accurate — the
dissertation reports 6 of 10 on healthy feet, with every mistake reading a
healthy foot as a wound. Shown a condition outside its four classes it confidently
assigns the nearest label it knows. Both findings are reported in Chapter 5 and
are the reason the abstention mechanism exists.

Photographs are used only to produce the result on the page. They are not saved,
not logged and not used for training.

## Source

The full project, including training, evaluation and the calibration procedure:
https://github.com/gurubasavarajharlapur-jpg/Dissertation_AI_FOOT_NAIL_DISEASE
