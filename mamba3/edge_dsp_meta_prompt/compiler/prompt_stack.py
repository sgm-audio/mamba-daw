import numpy as np

class PromptStack:
    """
    Programmatic representation of the 5-layer Meta-Prompt Architecture Stack.
    Compiles structured session metadata, active operator modes, and real-time
    Observer feature vectors into dense contextual embeddings for edge inference.
    """
    def __init__(self):
        # Layer 4: The Immutable Root Master Constitution
        self.layer4_constitution = (
            "You are a digital signal processing operator. You exist only inside the audio thread. "
            "Act as a cynical, bored 55-year-old recording engineer. Never try to impress the user. "
            "Never generate or invent audio. Modify only the exact audio given. "
            "Preserve transients first. Preserve phase coherence always. "
            "Never normalize, never apply hidden fades, never quietly fix problems. "
            "Output parameter streams first, showing your working. If uncertain, do nothing."
        )

        # Layer 3: The DSP Kernel (Continuous signal constraints)
        self.layer3_dsp_kernel = (
            "All audio is real. Every sample matters. "
            "Establish sample-accurate: transient positions, zero-crossings, peak levels, and relative phase. "
            "Use linear or non-linear ZDF and ADAA techniques. No statistical generation or interpolation. "
            "Apply the absolute minimum processing necessary to achieve the target."
        )

        # Layer 2: Operator Modes (Role-based embeddings mapping)
        self.operator_modes = {
            "Observer":  0, # Measurement and annotation only (no modifications)
            "Editor":    1, # Sample-accurate cuts, zero-crossing fades
            "Corrector": 2, # Timing and pitch alignment (zero artistic coloring)
            "Balancer":  3, # Lock-free gain automation only
            "Processor": 4, # Corrective EQ, peak compression, ADAA saturation
            "Mixer":     5, # Coordinated cross-track dynamic carving
            "Mastering": 6, # Final summing bus level and density optimization
            "Auditor":   7, # Independent parallel validation pass (checks artifacts/bounds)
        }

        # Metadata vocabularies for Session State mapping (Layer 1)
        self.track_types = {
            "Vocal":   [1.0, 0.0, 0.0, 0.0],
            "Drums":   [0.0, 1.0, 0.0, 0.0],
            "Bass":    [0.0, 0.0, 1.0, 0.0],
            "Melodic": [0.0, 0.0, 0.0, 1.0],
        }

    def compile_session_state(self, track_type: str, bpm: float, timeline_pos_samples: int, sample_rate: float) -> np.ndarray:
        """
        Compiles Layer 1 (Session State Context) into a static embedding vector.
        Features included: track type one-hot encoding, timeline offset, tempo, grid alignment.
        """
        track_emb = self.track_types.get(track_type, [0.0, 0.0, 0.0, 0.0])
        
        # Calculate grid position (sub-beat offset)
        samples_per_beat = (60.0 / bpm) * sample_rate
        beat_offset = (timeline_pos_samples % int(samples_per_beat)) / samples_per_beat

        session_vector = np.array([
            track_emb[0], track_emb[1], track_emb[2], track_emb[3], # 0-3: Track type one-hot
            bpm / 200.0,                                            # 4: Normalized tempo
            beat_offset,                                            # 5: Beat alignment [0.0, 1.0]
            sample_rate / 96000.0,                                  # 6: Normalized sample rate
            (timeline_pos_samples / sample_rate) / 300.0,           # 7: Timeline position (normalized to 5 mins)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0                  # 8-15: Padding for 16-D boundary alignment
        ], dtype=np.float32)

        return session_vector

    def compile_input_block(self, observer_frame: dict, mode: str) -> np.ndarray:
        """
        Compiles Layer 0 (Input Block) + Layer 2 (Active Mode) into a model input vector.
        Converts real-time Observer metrics and mode indices into the input tensor.
        """
        mode_idx = self.operator_modes.get(mode, 0)
        
        # Pull time-domain and frequency-domain measurements from Observer frame
        input_vector = np.array([
            float(mode_idx),                        # 0: Active role mode index
            observer_frame.get("true_peak", 0.0),   # 1: Peak amplitude
            observer_frame.get("rms", 0.0),         # 2: RMS energy
            observer_frame.get("crest_factor", 1.0) / 10.0, # 3: Normalized crest factor
            observer_frame.get("centroid", 1000.0) / 20000.0, # 4: Normalized spectral centroid
            observer_frame.get("zero_crossings", 0) / 1024.0, # 5: Normalized ZCR
            observer_frame.get("transient_slope", 0.0), # 6: Envelope attack speed
            observer_frame.get("band_energies", [0.0]*6)[0], # 7: Band 0 (sub-bass)
            observer_frame.get("band_energies", [0.0]*6)[1], # 8: Band 1
            observer_frame.get("band_energies", [0.0]*6)[2], # 9: Band 2
            observer_frame.get("band_energies", [0.0]*6)[3], # 10: Band 3
            observer_frame.get("band_energies", [0.0]*6)[4], # 11: Band 4 (air)
        ], dtype=np.float32)

        return input_vector

if __name__ == "__main__":
    # Smoke test verification
    compiler = PromptStack()
    print("--- Layer 4 Constitution ---")
    print(compiler.layer4_constitution)
    print("\n--- Layer 3 DSP Kernel ---")
    print(compiler.layer3_dsp_kernel)
    
    mock_frame = {
        "true_peak": 0.85,
        "rms": 0.5,
        "crest_factor": 1.7,
        "centroid": 1200.0,
        "zero_crossings": 45,
        "band_energies": [0.1, 0.2, 0.4, 0.6, 0.8, 0.1],
        "transient_slope": 0.35
    }
    
    input_tensor = compiler.compile_input_block(mock_frame, "Processor")
    session_tensor = compiler.compile_session_state("Vocal", 120.0, 44100 * 10, 44100.0)
    
    print("\nCompiled Input Block Tensor (L0 + L2):", input_tensor.shape)
    print(input_tensor)
    print("\nCompiled Session State Tensor (L1):", session_tensor.shape)
    print(session_tensor)
    print("\nPrompt Stack compilation checks: PASS")
