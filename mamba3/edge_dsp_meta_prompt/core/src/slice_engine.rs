/// Represents a segmented waveform fragment bounded by zero-crossings or maximum buffer limits.
/// Hard real-time safe: lightweight descriptor containing raw offsets into the circular buffer,
/// avoiding heap allocations or vector copies when passing work to the background thread.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SliceDescriptor {
    pub start_sample_abs: u64, // Absolute timeline playhead index at slice start
    pub length: u32,           // Length of the slice in samples
    pub zero_crossing_count: u32,
}

/// A zero-allocation, hard real-time safe waveform segmentation engine.
/// Continuously tracks zero-crossings and groups samples into descriptors.
pub struct SliceEngine {
    max_slice_len: usize,
    prev_sample: f32,
    abs_sample_counter: u64,
    last_zero_crossing: u64,
    zero_crossing_count: u32,
}

impl SliceEngine {
    /// Creates a new SliceEngine.
    /// Allocates heap memory. Must be called in init/constructor, never in the audio loop.
    pub fn new(max_slice_len: usize) -> Self {
        Self {
            max_slice_len,
            prev_sample: 0.0,
            abs_sample_counter: 0,
            last_zero_crossing: 0,
            zero_crossing_count: 0,
        }
    }

    /// Resets the slice engine state.
    pub fn reset(&mut self) {
        self.prev_sample = 0.0;
        self.abs_sample_counter = 0;
        self.last_zero_crossing = 0;
        self.zero_crossing_count = 0;
    }

    /// Processes a single incoming audio sample.
    /// Hard real-time safe: non-blocking, zero-allocation, O(1).
    /// Returns `Some(SliceDescriptor)` when a zero-crossing boundary is intersected
    /// or when the buffer length reaches the maximum threshold.
    #[inline(always)]
    pub fn process_sample(&mut self, sample: f32) -> Option<SliceDescriptor> {
        let current_index = self.abs_sample_counter;
        self.abs_sample_counter += 1;

        // Detect zero-crossing event (sign change)
        let is_zero_crossing = (sample >= 0.0 && self.prev_sample < 0.0)
            || (sample < 0.0 && self.prev_sample >= 0.0);

        self.prev_sample = sample;

        if is_zero_crossing {
            self.zero_crossing_count += 1;
        }

        let current_slice_len = current_index - self.last_zero_crossing;

        // Trigger condition: we crossed zero, or we hit the maximum allowed slice length
        if (is_zero_crossing && current_slice_len >= 64) || current_slice_len >= self.max_slice_len as u64 {
            let descriptor = SliceDescriptor {
                start_sample_abs: self.last_zero_crossing,
                length: current_slice_len as u32,
                zero_crossing_count: self.zero_crossing_count,
            };

            // Update indices for next slice cycle
            self.last_zero_crossing = current_index;
            self.zero_crossing_count = 0;

            Some(descriptor)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_segmentation() {
        let max_len = 512;
        let mut engine = SliceEngine::new(max_len);

        // Sinusoid that crosses zero every 50 samples
        let mut slices = Vec::new();
        for i in 0..1000 {
            // crosses zero when sin(pi * i / 50) = 0 => i is multiple of 50
            let sample = (std::f32::consts::PI * i as f32 / 50.0).sin();
            if let Some(slice) = engine.process_sample(sample) {
                slices.push(slice);
            }
        }

        // Each slice should correspond to the zero-crossing intervals
        assert!(!slices.is_empty());
        for slice in &slices {
            assert!(slice.length >= 50); // Minimum crossing length
            assert!(slice.zero_crossing_count > 0);
        }
    }
}
