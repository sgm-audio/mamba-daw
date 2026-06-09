use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A high-performance, real-time safe Single-Producer Single-Consumer (SPSC) Lock-Free Ring Buffer.
/// Implements cache-line padding to prevent false sharing and uses strict Acquire-Release atomic fences
/// to coordinate lock-free data transfer between the real-time audio thread and background workers.
pub struct SpscRingBuffer<T> {
    write_ptr: CachePaddedAtomic,
    read_ptr: CachePaddedAtomic,
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    capacity: usize,
}

#[repr(align(64))]
struct CachePaddedAtomic(AtomicUsize);

unsafe impl<T: Send> Send for SpscRingBuffer<T> {}
unsafe impl<T: Send> Sync for SpscRingBuffer<T> {}

impl<T> SpscRingBuffer<T> {
    /// Creates a new SpscRingBuffer with the requested capacity.
    /// Allocates memory on the heap. Must be called in init/constructor, never in the audio loop.
    pub fn new(capacity: usize) -> Self {
        let real_capacity = capacity + 1; // 1 extra slot to distinguish between full and empty
        let mut buffer = Vec::with_capacity(real_capacity);
        for _ in 0..real_capacity {
            buffer.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        Self {
            write_ptr: CachePaddedAtomic(AtomicUsize::new(0)),
            read_ptr: CachePaddedAtomic(AtomicUsize::new(0)),
            buffer: buffer.into_boxed_slice(),
            capacity: real_capacity,
        }
    }

    /// Pushes an item into the buffer.
    /// Hard real-time safe: non-blocking, zero-allocation, O(1).
    /// Returns `Err(item)` if the buffer is full.
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), T> {
        let write = self.write_ptr.0.load(Ordering::Relaxed);
        let read = self.read_ptr.0.load(Ordering::Acquire); // Ensure reader updates are visible
        let next_write = (write + 1) % self.capacity;

        if next_write == read {
            return Err(item); // Buffer is full
        }

        unsafe {
            let slot = self.buffer[write].get();
            *slot = MaybeUninit::new(item);
        }

        // Release ordering guarantees that the item write is visible before write_ptr updates
        self.write_ptr.0.store(next_write, Ordering::Release);
        Ok(())
    }

    /// Pops an item from the buffer.
    /// Hard real-time safe: non-blocking, zero-allocation, O(1).
    /// Returns `None` if the buffer is empty.
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let read = self.read_ptr.0.load(Ordering::Relaxed);
        let write = self.write_ptr.0.load(Ordering::Acquire); // Ensure writer updates are visible

        if read == write {
            return None; // Buffer is empty
        }

        let item = unsafe {
            let slot = self.buffer[read].get();
            std::ptr::read(slot).assume_init()
        };

        let next_read = (read + 1) % self.capacity;
        
        // Release ordering guarantees that slot read is finished before read_ptr updates
        self.read_ptr.0.store(next_read, Ordering::Release);
        Some(item)
    }

    /// Checks if the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let read = self.read_ptr.0.load(Ordering::Relaxed);
        let write = self.write_ptr.0.load(Ordering::Relaxed);
        read == write
    }

    /// Returns the number of items currently in the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        let read = self.read_ptr.0.load(Ordering::Relaxed);
        let write = self.write_ptr.0.load(Ordering::Relaxed);
        if write >= read {
            write - read
        } else {
            self.capacity - (read - write)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_push_pop() {
        let buffer = SpscRingBuffer::new(3);
        assert!(buffer.push(10).is_ok());
        assert!(buffer.push(20).is_ok());
        assert!(buffer.push(30).is_ok());
        assert!(buffer.push(40).is_err()); // Full (capacity is 3)

        assert_eq!(buffer.pop(), Some(10));
        assert_eq!(buffer.pop(), Some(20));
        assert_eq!(buffer.pop(), Some(30));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_thread_safety() {
        let buffer = std::sync::Arc::new(SpscRingBuffer::new(100));
        let buffer_clone = buffer.clone();

        let producer = thread::spawn(move || {
            for i in 0..1000 {
                while buffer_clone.push(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            for i in 0..1000 {
                let mut val = None;
                while val.is_none() {
                    val = buffer.pop();
                    if val.is_none() {
                        thread::yield_now();
                    }
                }
                assert_eq!(val.unwrap(), i);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
