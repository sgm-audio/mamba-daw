use std::sync::Arc;
use std::sync::Mutex;
use edge_dsp_core::TripleBuffer;
use crate::plugin::DynamicDspParams;

/// Log entry struct representing an audited edit committed to the history stack.
#[derive(Clone, Debug)]
pub struct OperationLogEntry {
    pub timestamp_ms: u64,
    pub action: String,
    pub confidence: f32,
}

/// The Parallel Auditor Agent. Intercepts parameters predicted by the Mamba3 controller,
/// evaluates them against the Root Constitution, clamps unstable boundaries (phase/clipping),
/// writes accepted parameters to the RT Triple-Buffer, and logs historical modifications.
pub struct AuditorAgent {
    param_sender: Arc<TripleBuffer<DynamicDspParams>>,
    
    // Mutex-guarded operations history log (shared with trace/UI threads)
    pub operation_log: Arc<Mutex<Vec<OperationLogEntry>>>,
}

impl AuditorAgent {
    /// Creates a new AuditorAgent instance.
    pub fn new(param_sender: Arc<TripleBuffer<DynamicDspParams>>) -> Self {
        Self {
            param_sender,
            operation_log: Arc::new(Mutex::new(Vec::with_capacity(1024))),
        }
    }

    /// Verifies the raw proposed parameters against the absolute safety rules.
    /// Hard real-time safe: runs on the worker thread, lock-free param writing.
    pub fn verify_and_commit(&self, mut params: DynamicDspParams) {
        let mut changes_logged = Vec::new();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // 1. Rule A: Phase Warp Protection
        // Wide SVF filters are stable, but extremely narrow high EQ gains cause phase group delay smear.
        // We clamp EQ gains to prevent clipping distortion or severe phase cancellation.
        for b in 0..6 {
            let max_gain_clamp = 9.0; // Dynamic EQ gains capped at +/- 9.0 dB
            if params.eq_gains_db[b] > max_gain_clamp {
                params.eq_gains_db[b] = max_gain_clamp;
                changes_logged.push(format!("EQ Band {} boost clamped to +9.0 dB (Phase safety limit)", b));
            } else if params.eq_gains_db[b] < -max_gain_clamp {
                params.eq_gains_db[b] = -max_gain_clamp;
                changes_logged.push(format!("EQ Band {} cut clamped to -9.0 dB (Phase safety limit)", b));
            }
        }

        // 2. Rule B: Over-Compression Safety
        // Heavy compression ratios (> 6.0) combined with deep thresholds cause unnatural pumping.
        if params.comp_ratio > 5.0 && params.comp_threshold_db < -30.0 {
            params.comp_ratio = 4.0;
            changes_logged.push("Compressor Ratio scaled back to 4.0 (Over-compression threshold alert)".to_string());
        }

        // 3. Rule C: Saturation Ceiling Guard
        // Polynomial saturators with > 8.0 dB drive produce heavy aliasing despite ADAA.
        if params.sat_drive_db > 8.0 {
            params.sat_drive_db = 8.0;
            changes_logged.push("ADAA Saturator drive clamped to 8.0 dB (Nyquist protection active)".to_string());
        }

        // 4. Commit verified parameters to the Real-Time thread (Triple-Buffer swap)
        self.param_sender.write(params);

        // 5. Log all committed corrections to the operation registry history stack
        if !changes_logged.is_empty() {
            if let Ok(mut log) = self.operation_log.lock() {
                // Keep the log length bounded to prevent memory exhaust
                if log.len() >= 1000 {
                    log.drain(0..100);
                }
                for change in changes_logged {
                    log.push(OperationLogEntry {
                        timestamp_ms: timestamp,
                        action: change,
                        confidence: 0.95, // 95% Auditor confidence score
                    });
                }
            }
        }
    }

    /// Fetches the entire operations log.
    pub fn get_log(&self) -> Vec<OperationLogEntry> {
        if let Ok(log) = self.operation_log.lock() {
            log.clone()
        } else {
            Vec::new()
        }
    }
}
