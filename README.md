# Naruto Clone Jutsu

This is a browser project where your webcam gesture triggers a Naruto-style shadow clone effect. It is basically a mix of computer vision, canvas tricks, and pure anime energy.

## Quick guide to this README
| Section | What it is for |
|---|---|
| Project in one line | Fast summary of what this app does |
| Tech stack | Tools and libraries used |
| Why it feels simple | Parts that were easy to build |
| Problems we fought | Real issues faced during development |
| Credits | Inspiration and asset source |

## Project in one line
Webcam + hand gesture detection + segmentation + clone effects + smoke + sound.

## Files Included
| File | Description |
|---|---|
| `index.html` | Main app: webcam feed with clone jutsu effect |
| `script.js` | Clone rendering, gesture detection, smoke effects, sound trigger |
| `style.css` | Styling for the main page |
| `trainer.html` | UI for recording hand sign samples and training |
| `trainer.js` | Training logic and model definition using hand landmarks |
| `trainer.css` | Styling for the trainer page |
| `assets/` | Smoke sprites, background image, overlay button images, sound asset |

## Tech stack
- HTML, CSS, JavaScript
- TensorFlow.js
- MediaPipe Holistic
- MediaPipe Selfie Segmentation
- Canvas 2D API

## Why it feels simple
- Single-page setup
- Direct webcam-to-canvas rendering
- Lightweight UI with visual effect assets

## Problems we fought
- Gesture confidence instability across lighting and camera angles
- Browser autoplay limits for sound playback
- Tuning trigger thresholds to avoid false positives and missed detections

## Credits
Assets and original idea inspiration:
https://github.com/nasha-wanich/naruto-shadow-clone-jutsu
