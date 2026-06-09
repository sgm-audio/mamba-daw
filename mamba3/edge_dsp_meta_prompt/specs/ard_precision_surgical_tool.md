# Architecture Reference Document (ARD)
## Concept C: The Transient-Coherent De-Esser & De-Clicker ("The Precision Surgical Tool")

---

## 1. Executive Technical Summary

This document specifies the concrete system and mathematical architecture for **Concept C: The Precision Surgical Tool**. This ARD outlines how the zero-crossing slice engine, waveform classification pipeline, and sample-accurate edit scheduler are implemented in a hard real-time safe CLAP execution loop.

---

## 2. Mathematical & Waveform-Surgery Foundations

### 2.1 Raised-Cosine Zero-Crossing aligned Micro-Fades

To prevent clicking and preserve the high-frequency spectral phase, all waveform surgery (de-clicking and muting pops) is performed strictly at zero-crossing boundaries utilizing a raised-cosine fade window to ensure $C^1$ continuity (continuous first derivative).

Given:
*   $n_{\text{start}}$: The sample index of the zero-crossing prior to the click.
*   $n_{\text{end}}$: The sample index of the zero-crossing after the click.
*   $L_{\text{fade}}$: The length of the fade window in samples ($L_{\text{fade}} = \lfloor f_s \cdot t_{\text{fade}} \rfloor$, where $t_{\text{fade}} \approx 1\text{ ms}$).

The fade-out envelope $w_{\text{out}}(k)$ for $k \in [0, L_{\text{fade}}-1]$ starting at $n_{\text{start}}$ is:
$$w_{\text{out}}(k) = \frac{1}{2} \left[ 1 + \cos\left(\frac{\pi k}{L_{\text{fade}}}\right) \right]$$

The fade-in envelope $w_{\text{in}}(k)$ for $k \in [0, L_{\text{fade}}-1]$ ending at $n_{\text{end}}$ is:
$$w_{\text{in}}(k) = \frac{1}{2} \left[ 1 - \cos\left(\frac{\pi k}{L_{\text{fade}}}\right) \right]$$

The intermediate region between the fades ($n_{\text{start}} + L_{\text{fade}} \le n < n_{\text{end}} - L_{\text{fade}}$) is fully attenuated (dry signal multiplied by $0.0$ or dynamically interpolated).

```
       Input Waveform:
       ───/\───[CLICK]───/\───────────
          │             │
          ▼ (n_start)   ▼ (n_end)
       Micro-Fade Window:
       ───\_____________/...
          ▲     ▲     ▲     ▲
          │     │     │     └─ Fade-In (w_in, L_fade)
          │     │     └─────── Attenuated region
          │     └───────────── Fade-Out (w_out, L_fade)
          └─────────────────── Zero crossing boundary
```

This guarantees that the signal slope matches the zero crossing exactly at the boundaries, eliminating high-frequency switching clicks.

### 2.2 Sibilance Vowel-Ratio Classifier

To accurately trigger the Dynamic EQ for de-essing, the classifier calculates the high-frequency sibilance ratio $R(n)$ over a moving window of $10\text{ ms}$:
$$R(n) = \frac{\sum_{k=f_1}^{f_s/2} |X(k)|^2}{\sum_{k=0}^{f_1} |X(k)|^2}$$
Where $f_1 = 4500\text{ Hz}$ represents the sibilance boundary, and $X(k)$ is the short-term FFT of the current slice.

If $R(n) > \theta_{\text{sib}}$ (where $\theta_{\text{sib}}$ is dynamically estimated based on the long-term noise floor), the slice is classified as sibilant. The dynamic EQ is steered to match the harmonic vowel's energy to prevent over-de-essing (lisping).

---

## 3. The Immutable Operation Log & Event Scheduler

All waveform edits are written to a sample-accurate, lock-free **Immutable Operation Log** which tracks the history and guides the real-time DSP engine's playhead.

### 3.1 Event Registry Struct Layout

```rust
#[repr(C)]
pub enum EditType {
    Bypass,
    MicroFade { fade_out_len: u32, fade_in_len: u32 },
    DynamicGain { target_gain_db: f32, attack_ms: f32, release_ms: f32 },
}

#[repr(C)]
pub struct SurgicalEditEvent {
    pub event_id: u64,
    pub start_sample: u64, // Absolute session timeline sample start
    pub end_sample: u64,   // Absolute session timeline sample end
    pub confidence: f32,
    pub edit_type: EditType,
}

// Thread-safe ring buffer containing the active schedule
pub struct EditEventScheduler {
    pub events: [Option<SurgicalEditEvent>; 1024], // Pre-allocated timeline bounds
    pub read_idx: AtomicUsize,
    pub write_idx: AtomicUsize,
}
```

### 3.2 Dynamic Search & Playhead Execution (RT Thread)
1.  During each audio block callback, the RT thread retrieves its absolute playhead position from the CLAP host interface (`clap_event_transport`).
2.  It queries the `EditEventScheduler` to find if the current block bounds $[n_{\text{block\_start}}, n_{\text{block\_end}}]$ overlap with any registered `SurgicalEditEvent`.
3.  If an overlap is found, the RT thread extracts the sub-sample offsets and applies the `MicroFade` or `DynamicGain` envelopes directly to the buffer, with sub-sample interpolation.

---

## 4. Thread Topologies & Waveform Segmentation

The system uses a decoupled three-stage execution pipeline.

```
       [RT Audio Callback]
                │
                ├─► zero_crossing_detector ──► Segment slices (No allocations)
                │                                    │
                │                                    ▼ (Push Slice Pointer)
                │                           [SPSC Ring Buffer]
                │                                    │
                │                                    ▼
                │                           [Worker Thread Pool]
                │                           - Extract Slice Features
                │                           - Quantized Classifier (Mamba3)
                │                           - Plan Fades / Gains
                │                                    │
                │                                    ▼ (Commit Event)
                │                           [Surgical Edit Scheduler]
                │                                    │
                │                                    ▼
                └◄─ Apply sample-accurate edits ◄────┘
```

### 4.1 Zero-Allocation Slice Engine
*   The Observer maintains a circular ring buffer of raw input audio of size $2048$ samples ($46\text{ ms}$ window).
*   When a zero-crossing boundary is detected, the Observer does not allocate a new vector; instead, it pushes a lightweight slice descriptor `(start_offset, end_offset)` into the SPSC queue.
*   The Worker thread reads the raw audio directly from the mapped ring buffer using these offsets, ensuring **zero heap allocations** throughout the entire lifecycle.

---

## 5. Model Interface & Tensor Topologies

The surgical sequence classifier expects pre-processed slice vectors.

### 5.1 Classifier Tensor Signatures

#### Input Tensors
*   `slice_features`: Shape `[1, 16]` (containing zero-crossing density, spectral centroid, spectral flatness, RMS slope, high-to-low ratio, asymmetry metrics). Type: `Tensor<f32>`.
*   `slice_waveform`: Shape `[1, 1024]` (padded raw waveform segment). Type: `Tensor<f32>`.

#### Output Tensors
*   `classification_probabilities`: Shape `[1, 3]` (representing `[Clean, Sibilant, Click]` probabilities). Type: `Tensor<f32>`.
*   `planned_surgery`: Shape `[1, 4]` (containing recommended fade-out length, fade-in length, target attenuation, sibilant center frequency). Type: `Tensor<f32>`.

---

## 6. System Component Breakdown & Code Signatures

### 6.1 `core::slice_engine` Module
```rust
pub struct SliceDescriptor {
    pub start_sample_abs: u64,
    pub length: u32,
    pub zero_crossing_count: u32,
}

pub struct SliceEngine {
    raw_buffer: Vec<f32>,
    write_ptr: usize,
    prev_sample: f32,
    abs_sample_counter: u64,
    last_zero_crossing: u64,
}

impl SliceEngine {
    pub fn new(max_slice_len: usize) -> Self;
    
    // Hard real-time safe function
    #[inline(always)]
    pub fn process_sample(&mut self, sample: f32) -> Option<SliceDescriptor>;
}
```

### 6.2 `core::surgical_renderer` Module
```rust
pub struct SurgicalRenderer {
    active_fade_index: usize,
    fade_envelope: Vec<f32>,
}

impl SurgicalRenderer {
    pub fn new() -> Self;
    
    // Hard real-time safe function
    #[inline(always)]
    pub fn render_fade_out(&mut self, buffer: &mut [f32], start_idx: usize, length: usize);
    
    // Hard real-time safe function
    #[inline(always)]
    pub fn render_fade_in(&mut self, buffer: &mut [f32], start_idx: usize, length: usize);
}
```
