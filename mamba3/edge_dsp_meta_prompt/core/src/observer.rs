use std::sync::Arc;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

/// The extracted feature frame pushed by the Observer from the RT callback.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObserverFrame {
    pub true_peak: f32,
    pub rms: f32,
    pub crest_factor: f32,
    pub centroid: f32,
    pub zero_crossings: u32,
    pub band_energies: [f32; 6],
    pub transient_slope: f32,
}

/// A zero-allocation, hard real-time safe feature extractor.
/// Continuously tracks time-domain characteristics (peak, RMS, crest, transients)
/// and window-domain characteristics (FFT-based spectral centroid and critical log-Hz bands).
pub struct Observer {
    sample_rate: f32,
    block_size: usize,
    
    // Time domain envelope tracking
    rms_fast: f32,
    rms_slow: f32,
    peak_hold: f32,
    prev_sample: f32,
    zero_crossing_count: u32,
    sample_counter: usize,
    
    // FFT Analysis buffers (pre-allocated)
    input_buffer: Vec<f32>,
    output_buffer: Vec<Complex<f32>>,
    window: Vec<f32>,
    fft_plan: Arc<dyn RealToComplex<f32>>,
    fft_buffer_idx: usize,
    
    // Pre-computed filterbank bin boundaries
    band_bins: [FilterBankBand; 6],
}

struct FilterBankBand {
    start_bin: usize,
    center_bin: usize,
    end_bin: usize,
}

impl Observer {
    /// Creates a new Observer. Allocates heap vectors.
    /// Must be called during plugin instantiation, never in the audio loop.
    pub fn new(block_size: usize, sample_rate: f32) -> Self {
        // We use an FFT size equal to the block size or next power of 2 (e.g. 512, 1024 or 2048)
        let fft_size = block_size;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(fft_size);

        // Pre-allocate input and output complex buffers
        let input_buffer = fft_plan.make_input_vec();
        let output_buffer = fft_plan.make_output_vec();

        // Pre-compute Hann window to suppress spectral leakage
        let mut window = vec![0.0; fft_size];
        for i in 0..fft_size {
            window[i] = 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (fft_size - 1) as f32).cos());
        }

        // Pre-compute log-spaced triangular bands (from ~30 Hz to ~16 kHz)
        let f_min = 30.0;
        let f_max = 16000.0;
        let num_bands = 6;
        
        let mel_min = 1127.0 * (1.0 + f_min / 700.0).ln();
        let mel_max = 1127.0 * (1.0 + f_max / 700.0).ln();
        
        let mut band_bins = [
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
            FilterBankBand { start_bin: 0, center_bin: 0, end_bin: 0 },
        ];

        let bin_hz = sample_rate / fft_size as f32;

        for i in 0..num_bands {
            let mel_start = mel_min + (i as f32) * (mel_max - mel_min) / (num_bands as f32 + 1.0);
            let mel_center = mel_min + ((i + 1) as f32) * (mel_max - mel_min) / (num_bands as f32 + 1.0);
            let mel_end = mel_min + ((i + 2) as f32) * (mel_max - mel_min) / (num_bands as f32 + 1.0);

            let hz_start = 700.0 * ((mel_start / 1127.0).exp() - 1.0);
            let hz_center = 700.0 * ((mel_center / 1127.0).exp() - 1.0);
            let hz_end = 700.0 * ((mel_end / 1127.0).exp() - 1.0);

            band_bins[i] = FilterBankBand {
                start_bin: (hz_start / bin_hz).round() as usize,
                center_bin: (hz_center / bin_hz).round() as usize,
                end_bin: (hz_end / bin_hz).round() as usize,
            };
        }

        Self {
            sample_rate,
            block_size,
            rms_fast: 0.0,
            rms_slow: 0.0,
            peak_hold: 0.0,
            prev_sample: 0.0,
            zero_crossing_count: 0,
            sample_counter: 0,
            input_buffer,
            output_buffer,
            window,
            fft_plan,
            fft_buffer_idx: 0,
            band_bins,
        }
    }

    /// Processes a single incoming audio sample.
    /// Hard real-time safe: zero-allocation, lock-free, O(1) per sample.
    /// Returns `Some(ObserverFrame)` once a full block of samples has been collected.
    #[inline(always)]
    pub fn process_sample(&mut self, sample: f32) -> Option<ObserverFrame> {
        let abs_sample = sample.abs();
        
        // 1. Peak Envelope Tracking (Fast attack, slow decay)
        if abs_sample > self.peak_hold {
            self.peak_hold = abs_sample;
        } else {
            self.peak_hold *= 0.9995; // Leaky integration (~100ms decay)
        }

        // 2. RMS Envelope Tracking (Dual-time integration)
        let sample_sq = sample * sample;
        self.rms_fast = 0.05 * sample_sq + 0.95 * self.rms_fast; // Fast envelope (~10ms)
        self.rms_slow = 0.005 * sample_sq + 0.995 * self.rms_slow; // Slow envelope (~100ms)

        // 3. Zero Crossing Rate Tracking
        if (sample >= 0.0 && self.prev_sample < 0.0) || (sample < 0.0 && self.prev_sample >= 0.0) {
            self.zero_crossing_count += 1;
        }
        self.prev_sample = sample;

        // 4. Fill FFT analysis sliding buffer
        if self.fft_buffer_idx < self.block_size {
            // Apply windowing function directly on entry
            self.input_buffer[self.fft_buffer_idx] = sample * self.window[self.fft_buffer_idx];
            self.fft_buffer_idx += 1;
        }

        self.sample_counter += 1;

        // Once the block is full, execute spectral computations and return the feature frame
        if self.sample_counter >= self.block_size {
            let frame = self.compute_block_frame();
            
            // Reset block accumulation states
            self.sample_counter = 0;
            self.fft_buffer_idx = 0;
            self.zero_crossing_count = 0;
            
            Some(frame)
        } else {
            None
        }
    }

    /// Computes spectral features on block completion.
    /// Hard real-time safe: runs the pre-allocated forward real FFT. O(N log N).
    fn compute_block_frame(&mut self) -> ObserverFrame {
        // Execute the zero-allocation forward FFT in place
        self.fft_plan.process(&mut self.input_buffer, &mut self.output_buffer).unwrap();

        let fft_len = self.output_buffer.len(); // Equals block_size / 2 + 1
        let bin_hz = self.sample_rate / self.block_size as f32;

        let mut sum_mag = 0.0;
        let mut sum_centroid = 0.0;
        
        // 1. Calculate magnitudes and Spectral Centroid
        for i in 0..fft_len {
            let mag = self.output_buffer[i].norm(); // Compute complex magnitude
            sum_mag += mag;
            sum_centroid += (i as f32 * bin_hz) * mag;
        }

        let centroid = if sum_mag > 0.0 { sum_centroid / sum_mag } else { 0.0 };

        // 2. Calculate Triangular Mel-Log-Hz Band Energies
        let mut band_energies = [0.0; 6];
        for b in 0..6 {
            let band = &self.band_bins[b];
            let mut accum = 0.0;
            
            for bin in band.start_bin..=band.end_bin {
                if bin < fft_len {
                    let mag = self.output_buffer[bin].norm();
                    
                    // Apply triangular weight
                    let weight = if bin < band.center_bin {
                        if band.center_bin == band.start_bin { 1.0 } else {
                            (bin - band.start_bin) as f32 / (band.center_bin - band.start_bin) as f32
                        }
                    } else {
                        if band.end_bin == band.center_bin { 1.0 } else {
                            (band.end_bin - bin) as f32 / (band.end_bin - band.center_bin) as f32
                        }
                    };
                    
                    accum += mag * weight;
                }
            }
            // Logarithmic compression to mimic human loudness perception (dB scale)
            band_energies[b] = (1.0 + accum).ln();
        }

        let rms = self.rms_fast.sqrt();
        let crest_factor = if rms > 1e-5 { self.peak_hold / rms } else { 1.0 };
        
        // Dynamic envelope derivative (attack slope)
        let transient_slope = (self.rms_fast - self.rms_slow).max(0.0).sqrt();

        ObserverFrame {
            true_peak: self.peak_hold,
            rms,
            crest_factor,
            centroid,
            zero_crossings: self.zero_crossing_count,
            band_energies,
            transient_slope,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_processing() {
        let block_size = 512;
        let sample_rate = 44100.0;
        let mut observer = Observer::new(block_size, sample_rate);

        // Generate a 1 kHz sine wave
        let mut frame_opt = None;
        for n in 0..block_size {
            let sample = (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / sample_rate).sin();
            frame_opt = observer.process_sample(sample);
        }

        assert!(frame_opt.is_some());
        let frame = frame_opt.unwrap();
        
        // Assertions verifying that values are populated and within sensible limits
        assert!(frame.true_peak > 0.9);
        assert!(frame.rms > 0.6);
        assert!(frame.centroid > 800.0 && frame.centroid < 1200.0); // Close to 1000 Hz
        assert!(frame.crest_factor > 1.0);
    }
}
