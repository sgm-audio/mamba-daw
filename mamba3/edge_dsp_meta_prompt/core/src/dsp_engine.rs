/// A zero-delay feedback (ZDF) state-variable filter (SVF) based on bilinear transformation
/// (trapezoidal integration). Safe for real-time sample-rate sweeps and dynamic EQ automation.
/// Provides perfectly stable lowpass, highpass, bandpass, notch, allpass, and bell (peaking EQ) responses.
#[derive(Clone, Debug, Default)]
pub struct ZdfSvf {
    s1: f32, // State integrator 1 (Bandpass state)
    s2: f32, // State integrator 2 (Lowpass state)
}

pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    AllPass,
    Bell { gain_db: f32 },
}

impl ZdfSvf {
    /// Creates a new ZdfSvf.
    pub fn new() -> Self {
        Self { s1: 0.0, s2: 0.0 }
    }

    /// Resets the internal integrator states to zero.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Processes a single sample through the filter.
    /// Hard real-time safe: non-blocking, zero-allocation, O(1) complexity.
    #[inline(always)]
    pub fn process(&mut self, input: f32, cutoff_hz: f32, q_factor: f32, filter_type: &FilterType, sample_rate: f32) -> f32 {
        // Pre-warp cutoff frequency to map continuous-time s-plane to digital z-plane
        let cutoff_rad = std::f32::consts::PI * cutoff_hz / sample_rate;
        let g = cutoff_rad.tan(); // Analog-equivalent pre-warped integrator gain

        let (k, a) = match filter_type {
            FilterType::Bell { gain_db } => {
                let a_boost = 10.0f32.powf(*gain_db / 40.0);
                // Adjust damping factor based on boost or cut to maintain symmetry
                if *gain_db >= 0.0 {
                    (1.0 / (q_factor * a_boost), a_boost)
                } else {
                    (a_boost / q_factor, a_boost)
                }
            }
            _ => (1.0 / q_factor, 1.0),
        };

        // Solve the zero-delay algebraic loop to find the highpass state
        let denominator = 1.0 + k * g + g * g;
        let v_hp = (input - (k + g) * self.s1 - self.s2) / denominator;

        // Compute remaining band responses
        let v_bp = self.s1 + g * v_hp;
        let v_lp = self.s2 + g * v_bp;

        // Update internal integrator states (trapezoidal integration)
        self.s1 += 2.0 * g * v_hp - k * v_bp;
        self.s2 += 2.0 * g * v_bp;

        // Mix outputs based on requested filter type
        match filter_type {
            FilterType::LowPass => v_lp,
            FilterType::HighPass => v_hp,
            FilterType::BandPass => v_bp,
            FilterType::Notch => input - k * v_bp,
            FilterType::AllPass => input - 2.0 * k * v_bp,
            FilterType::Bell { gain_db } => {
                if *gain_db >= 0.0 {
                    // Boost: input + (A^2 - 1) * k * v_bp
                    input + (a * a - 1.0) * k * v_bp
                } else {
                    // Cut: input + (1/A^2 - 1) * k * v_bp
                    input + (1.0 / (a * a) - 1.0) * k * v_bp
                }
            }
        }
    }
}

/// A non-linear waveshaping saturator employing first-order Anti-Derivative Anti-Aliasing (ADAA).
/// Evaluates continuous integration on input boundaries to suppress high-frequency aliasing foldback.
#[derive(Clone, Debug, Default)]
pub struct AdaaSaturator {
    prev_input: f32,
}

impl AdaaSaturator {
    /// Creates a new AdaaSaturator.
    pub fn new() -> Self {
        Self { prev_input: 0.0 }
    }

    /// Resets the saturator history.
    pub fn reset(&mut self) {
        self.prev_input = 0.0;
    }

    /// Evaluates the cubic soft-clipping function: f(x) = x - 1/3 * x^3, bounded to [-2/3, 2/3].
    #[inline(always)]
    fn evaluate_f(&self, x: f32) -> f32 {
        if x > 1.0 {
            2.0 / 3.0
        } else if x < -1.0 {
            -2.0 / 3.0
        } else {
            x - (1.0 / 3.0) * x * x * x
        }
    }

    /// Evaluates the analytical antiderivative F(x) = \int f(x) dx.
    /// Aligned constant terms ensure perfect mathematical continuity across limits.
    #[inline(always)]
    fn evaluate_antiderivative_f(&self, x: f32) -> f32 {
        if x > 1.0 {
            (2.0 / 3.0) * x - 0.25
        } else if x < -1.0 {
            -(2.0 / 3.0) * x - 0.25
        } else {
            0.5 * x * x - (1.0 / 12.0) * x * x * x * x
        }
    }

    /// Processes a single sample through the ADAA saturator.
    /// Hard real-time safe: non-blocking, zero-allocation, O(1).
    #[inline(always)]
    pub fn process(&mut self, input: f32, drive_db: f32, mix: f32) -> f32 {
        // Apply drive gain to input signal
        let drive_gain = 10.0f32.powf(drive_db / 20.0);
        let driven_input = input * drive_gain;

        // Perform first-order ADAA
        let epsilon = 1e-5;
        let delta = driven_input - self.prev_input;

        let saturated = if delta.abs() > epsilon {
            // Apply integration over sample boundaries
            let num = self.evaluate_antiderivative_f(driven_input) - self.evaluate_antiderivative_f(self.prev_input);
            num / delta
        } else {
            // Midpoint fallback for numerical stability near delta = 0
            self.evaluate_f(0.5 * (driven_input + self.prev_input))
        };

        // Cache the current input for the next sample cycle
        self.prev_input = driven_input;

        // Dry/Wet mixing
        (1.0 - mix) * input + mix * saturated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svf_lowpass() {
        let mut svf = ZdfSvf::new();
        let fs = 44100.0;
        let fc = 1000.0;
        let q = 0.707;

        // Verify that a DC input is preserved by a lowpass filter
        let mut out = 0.0;
        for _ in 0..100 {
            out = svf.process(1.0, fc, q, &FilterType::LowPass, fs);
        }
        assert!((out - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_adaa_clipper() {
        let mut saturator = AdaaSaturator::new();
        
        // Heavily driven input (should clip)
        let out_clipped = saturator.process(5.0, 10.0, 1.0);
        assert!((out_clipped.abs() - 2.0 / 3.0).abs() < 1e-2);

        // Weak input (should remain linear)
        let out_linear = saturator.process(0.1, 0.0, 1.0);
        assert!((out_linear - 0.1).abs() < 5e-2);
    }
}
