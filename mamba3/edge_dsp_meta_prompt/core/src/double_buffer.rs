use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A lock-free, zero-allocation, real-time safe Triple Buffer for parameter exchange.
/// Allows a background worker thread (Producer) to continuously write parameter updates
/// and the real-time audio thread (Consumer) to always read the latest complete set of parameters
/// without blocking, introducing zero jitter or priority inversion hazards.
pub struct TripleBuffer<T> {
    buffers: [UnsafeCell<T>; 3],
    active_idx: AtomicUsize, // Owned by Consumer (RT Thread)
    pending_idx: AtomicUsize, // Shared atomic
    free_idx: AtomicUsize,    // Owned by Producer (Worker Thread)
    dirty: AtomicBool,        // Shared atomic flag indicating a new write is committed
}

unsafe impl<T: Send> Send for TripleBuffer<T> {}
unsafe impl<T: Send> Sync for TripleBuffer<T> {}

impl<T: Clone> TripleBuffer<T> {
    /// Creates a new TripleBuffer with initial data in all slots.
    /// Allocates heap memory. Must be called in init/constructor, never in the audio loop.
    pub fn new(initial: T) -> Self {
        Self {
            buffers: [
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial),
            ],
            active_idx: AtomicUsize::new(0),
            pending_idx: AtomicUsize::new(1),
            free_idx: AtomicUsize::new(2),
            dirty: AtomicBool::new(false),
        }
    }

    /// Writes a new item to the TripleBuffer.
    /// Executed by the background Worker thread (Producer).
    /// Non-blocking, zero-allocation, O(1).
    pub fn write(&self, item: T) {
        let free = self.free_idx.load(Ordering::Relaxed);
        
        // Write the data into the current private Free slot
        unsafe {
            let slot = self.buffers[free].get();
            *slot = item;
        }

        // Atomically publish the Free slot by swapping it with the shared Pending slot.
        // Release ordering guarantees that the write to buffers[free] is visible
        // to the consumer before pending_idx updates.
        let pending = self.pending_idx.swap(free, Ordering::Release);
        
        // The old pending slot now becomes our private Free slot for the next write
        self.free_idx.store(pending, Ordering::Relaxed);
        
        // Set the dirty flag to alert the consumer that new parameters are available
        self.dirty.store(true, Ordering::Release);
    }

    /// Reads the latest published data.
    /// Executed by the real-time audio thread (Consumer).
    /// Non-blocking, zero-allocation, lock-free, O(1).
    /// Always returns a reference to the latest complete parameter set.
    #[inline(always)]
    pub fn read(&self) -> &T {
        // Check if there is a new commit
        // Acquire ordering guarantees visibility of the write to buffers[pending]
        if self.dirty.swap(false, Ordering::Acquire) {
            let active = self.active_idx.load(Ordering::Relaxed);
            
            // Swap our Active slot with the shared Pending slot
            let pending = self.pending_idx.swap(active, Ordering::AcqRel);
            
            // Store the new Active index locally
            self.active_idx.store(pending, Ordering::Relaxed);
            
            unsafe { &*self.buffers[pending].get() }
        } else {
            // No new write, read from the current active slot
            let active = self.active_idx.load(Ordering::Relaxed);
            unsafe { &*self.buffers[active].get() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[derive(Clone, Debug, PartialEq)]
    struct Params {
        gain: f32,
        cutoff: f32,
    }

    #[test]
    fn test_triple_buffer_basic() {
        let initial = Params { gain: 1.0, cutoff: 1000.0 };
        let buffer = TripleBuffer::new(initial.clone());

        // Initial read
        assert_eq!(buffer.read(), &initial);

        // Write new parameters
        let p1 = Params { gain: 0.8, cutoff: 1200.0 };
        buffer.write(p1.clone());

        // Read updated parameters
        assert_eq!(buffer.read(), &p1);
        
        // Sequential read with no new write should return same parameters
        assert_eq!(buffer.read(), &p1);
    }

    #[test]
    fn test_concurrent_read_write() {
        let initial = Params { gain: 1.0, cutoff: 1000.0 };
        let buffer = std::sync::Arc::new(TripleBuffer::new(initial));
        let buffer_clone = buffer.clone();

        let producer = thread::spawn(move || {
            for i in 0..1000 {
                buffer_clone.write(Params {
                    gain: i as f32,
                    cutoff: (1000 + i) as f32,
                });
                thread::sleep(Duration::from_micros(10));
            }
        });

        let consumer = thread::spawn(move || {
            let mut last_gain = -1.0;
            for _ in 0..2000 {
                let params = buffer.read();
                assert!(params.gain >= last_gain); // Strictly monotonic updates
                last_gain = params.gain;
                thread::sleep(Duration::from_micros(5));
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
