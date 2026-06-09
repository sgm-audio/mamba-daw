# Product Requirements Document (PRD)
## Concept C: The Transient-Coherent De-Esser & De-Clicker ("The Precision Surgical Tool")

---

## 1. Executive Summary

### 1.1 Product Vision
"The Precision Surgical Tool" is a sample-accurate corrective CLAP plugin designed to isolate and eliminate vocal sibilance ("S" sounds), mouth clicks, digital pops, and plosives, without altering the surrounding high-fidelity performance. Unlike conventional threshold-based de-essers or de-clickers that apply a generic compression band or low-pass filter over the entire high frequency spectrum, this tool utilizes an **asynchronous waveform-surgery model**. It detects audio artifacts at the zero-crossings, generates micro-adjustments in the background using a quantized classification model, and renders these corrections using **sub-millisecond micro-fades and sample-accurate gain envelopes** directly inside the audio thread.

### 1.2 The Problem Statement
Classical de-essers apply a frequency-selective compressor over the $4\text{ kHz}$ to $10\text{ kHz}$ band. When the threshold is triggered, the *entire* band is attenuated, removing desirable breathiness, clarity, and articulation. Traditional de-clickers interpolate over clicks blindly, leaving audible "dimples" or phase holes in the waveform that degrade high-end transparency.

### 1.3 The Solution: Zero-Crossing Audio Surgery
By mapping out Zero-Crossing Boundaries and Transient Onsets in the hard real-time thread, the plugin exposes precise surgical edit points. The background **Surgical AI** evaluates each isolated block. If an artifact (mouth click or sibilance) is classified, it writes a custom corrective sample-gain curve into the **Immutable Operation Log**, which the real-time DSP engine applies with sub-millisecond, sample-accurate precision.

```
       [Input Waveform] ─────────────► [Zero-Crossing Detector]
                                                │
                                                ▼ (Isolate Slice)
  ┌─────────────────────────────────────────────┴────────────────────────┐
  │                        SURGICAL CLASSIFICATION THREAD                │
  │                                                                      │
  │   [Audio Slice] ──► [Feature Extraction] ──► [Mamba3 Classifier]     │
  │                                                    │                 │
  │                                                    ▼ (Output: Click) │
  │                                            [Select Strategy]         │
  │                                            - Gain Ducking (Sibilant) │
  │                                            - Micro-Fade (Click)      │
  │                                                    │                 │
  └────────────────────────────────────────────────────┼─────────────────┘
                                                       ▼ (Write)
  ┌──────────────────────────────────────────────────────────────────────┐
  │                       IMMUTABLE OPERATION LOG                        │
  │                                                                      │
  │    Edit Stack: Slice 1204 ──► Apply 0.8ms fade-out, 1.2ms fade-in    │
  │                             At Zero Crossings [n_start, n_end]       │
  └────────────────────────────────────────────────────┬─────────────────┘
                                                       ▼ (Render)
  ┌────────────────────────────────────────────────────┴─────────────────┐
  │                         HARD REAL-TIME AUDIO THREAD                  │
  │                                                                      │
  │   [Dry Waveform] ──► [Gain/Fade DSP Engine] ──► [Surgically Cleaned]  │
  └────────────────────────────────────────────────────┘
```

---

## 2. Zero-Crossing & Slice Isolation Specifications

### 2.1 Waveform Segmentation (Slice Engine)
*   **Zero-Crossing Detector**: The real-time thread continuously tracks zero-crossing boundaries (samples where the amplitude changes sign).
*   **Slice Definition**: A "slice" is defined as the waveform fragment bounded by two zero-crossings containing a positive or negative half-cycle (or a series of zero-crossings spanning up to $20\text{ ms}$).
*   **Transient Onset Detection**: Utilizes high-frequency energy ratio variations to detect transient boundaries, segmenting transient bursts separately from continuous vocal sustain.

---

## 3. Surgical AI Specifications (Model Layer 1 & 2)

The background model is trained specifically as a **Sequence Classifier & Path Planner**.

### 3.1 Feature Vector Specs
For each isolated slice, the Observer extracts:
*   **Zero Crossing Density**: Frequency of crossings inside the $10\text{ ms}$ window.
*   **High-to-Low Ratio**: Energy ratio between $f > 5\text{ kHz}$ and $f < 2\text{ kHz}$.
*   **Phase Symmetry**: Difference in area between the positive and negative cycles of the slice.
*   **Envelope Peak-to-Average Ratio**: Ratio indicating whether the energy is transient (click) or continuous (sibilant).

### 3.2 Classification Matrix
The model classifies each slice into one of three states:

| Class ID | Spectral Profile | Target Action |
| :--- | :--- | :--- |
| **Clean Audio** | Harmonic structure, low noise-to-signal ratio, stable phase. | **Bypass**: Maintain $1.0$ gain, no editing. |
| **Sibilant ("S")** | High noise-to-signal ratio, concentrated in $4\text{ kHz}$ to $12\text{ kHz}$, no harmonic poles. | **Steerable Dynamic EQ / Gain**: Apply smooth high-frequency attenuation envelope. |
| **Click / Pop** | High energy peak spanning $<2\text{ ms}$ with highly asymmetric zero-crossing patterns. | **Micro-Fade Surgery**: Apply zero-crossing aligned fade-out and fade-in envelopes. |

---

## 4. Waveform Surgery Execution Specs (DSP Engine)

The real-time DSP engine applies the surgical edits strictly at zero-crossing boundaries, completely eliminating clicking or discontinuity artifacts.

### 4.1 De-Clicking: Micro-Fade Surgery
When a pop or click is detected between samples $n_{\text{start}}$ and $n_{\text{end}}$:
1.  **Locate Zero-Crossings**: Expand the search boundary slightly to find the nearest surrounding zero-crossing samples $n_{\text{zc\_left}}$ and $n_{\text{zc\_right}}$.
2.  **Generate Fade Envelopes**: Create a linear or raised-cosine fade-out envelope starting at $n_{\text{zc\_left}}$ and a matched fade-in envelope ending at $n_{\text{zc\_right}}$.
3.  **Cross-Fade Window**: Keep the fade length extremely short ($0.5\text{ ms}$ to $2.0\text{ ms}$) to remove only the click while leaving the surrounding signal mathematically untouched.

### 4.2 De-Essing: Steerable Dynamic EQ
When sibilance is classified:
1.  **Calculate Target Reduction**: Predict the required attenuation (typically $-3\text{ dB}$ to $-9\text{ dB}$) to balance the sibilant with the preceding harmonic vowel.
2.  **Zero-Delay Crossover**: Dynamic EQ bands are engaged instantly, utilizing zero-delay SVF filters.
3.  **Smooth Transition**: The gain reduction coefficient is ramped smoothly at zero-crossings to prevent high-frequency transients from clicking during filter activation.

---

## 5. Auditor and Safety Guardrails

Surgical edits can destroy natural articulation (e.g. turning "S" sounds into "F" sounds, known as "lisping") if over-applied. The **Auditor** acts as the clinical validator.

### 5.1 Anti-Lisping Protections
*   **Rule**: The system must never reduce sibilance to the point where the sibilant's spectral centroid drops below the classification boundary for an "F" or "Th" sound.
*   **Action**: If the proposed de-esser gain reduction exceeds the sibilant threshold, the Auditor clamps the gain reduction to a maximum of $-8.5\text{ dB}$.

### 5.2 De-Clicking Density Limit
*   **Rule**: To prevent phase holes, the de-clicker must never apply micro-fade surgery to more than 3 distinct slices within a continuous $100\text{ ms}$ window.
*   **Action**: If click density exceeds this limit (which indicates a highly noisy or corrupted file rather than discrete clicks), the Auditor disables micro-fade surgery and falls back to a gentle high-frequency dynamic shelf.

---

## 6. UI/UX & Waveform Surgery View

### 6.1 The Waveform Surgery Canvas
The center of the UI displays a scrolling waveform of the input audio. Detected artifacts are highlighted with bright HSL tailored colors (e.g., cool cyan for clicks, and warm amber for sibilants):

```
     Waveform Display:
     ───────/\───/\───────────/\──────────────/\─────────
            │     │           │ (Highlighted Sibilance)
            ▼     ▼           ▼
         [Click] [Click]   [De-Esser Gain: -4.5dB]
```

*   **Interactive Playback**: The user can hover their mouse over any highlighted click or sibilant in the scrollbar and click to **Bypass the edit**, **Listen only to the removed artifact (Dry minus Wet delta)**, or **Manually adjust the surgical width**.

### 6.2 User Controls

| Control | Range | Description |
| :--- | :--- | :--- |
| **De-Click Intensity** | 0% to 100% | Adjusts the detection sensitivity and width of the micro-fade surgical window. |
| **De-Ess Intensity** | 0% to 100% | Globally scales the depth of high-frequency attenuation during classified sibilance. |
| **Surgery Mode** | Safe / Clinical / Pure Manual | Sets the Auditor's enforcement level (Safe = high limits, Clinical = surgical precision, Manual = user overrides all). |
