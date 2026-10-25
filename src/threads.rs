//! A tiny fork-join helper built on `std::thread::scope`.
//!
//! `par_range` splits `0..n` into contiguous chunks (at least `min_chunk`
//! wide) and runs `f(start, end)` on each from a fresh thread. Below the
//! threshold it runs inline on the calling thread, which keeps small layers
//! allocation- and spawn-free. No worker pool: at these workload sizes spawn
//! cost is noise next to a 500 MB weight stream.

use std::marker::PhantomData;
use std::thread;

pub fn worker_count() -> usize {
    // NIMBUS_THREADS overrides the default (all logical cores). Useful on
    // hybrid CPUs where pinning to the P-cores can be faster for
    // memory-bound workloads.
    if let Ok(n) = std::env::var("NIMBUS_THREADS") {
        if let Ok(n) = n.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Run `f(start, end)` over `[0, n)` in parallel chunks with a worker cap.
pub fn par_range_capped<F>(n: usize, min_chunk: usize, max_workers: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    let workers = worker_count().min(max_workers);
    if n <= min_chunk || workers <= 1 {
        f(0, n);
        return;
    }
    let target_chunks = workers * 2;
    let chunk = ((n + target_chunks - 1) / target_chunks).max(min_chunk);
    let f = &f;
    thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + chunk).min(n);
            handles.push(scope.spawn(move || f(start, end)));
            start = end;
        }
        for handle in handles {
            handle.join().expect("worker thread panicked");
        }
    });
}

/// Run `f(start, end)` over `[0, n)` in parallel chunks.
pub fn par_range<F>(n: usize, min_chunk: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    par_range_capped(n, min_chunk, usize::MAX, f)
}

/// A shared mutable slice handed to fork-join workers that write **disjoint**
/// index ranges. This is the hand-rolled equivalent of rayon's
/// `par_chunks_mut`: the caller (the matmul kernels) guarantees each index is
/// produced by exactly one worker, so the aliasing is never observed.
pub struct UnsafeSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<'a, T: Send> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> UnsafeSlice<'a, T> {
        UnsafeSlice {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    /// Index `i` must be written by exactly one thread during the fan-out.
    #[inline]
    pub unsafe fn write(&self, i: usize, value: T) {
        debug_assert!(i < self.len);
        unsafe { *self.ptr.add(i) = value };
    }

    /// # Safety
    /// Indices in `range` must be written by exactly one thread.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut(&self, range: std::ops::Range<usize>) -> &mut [T] {
        debug_assert!(range.end <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn covers_every_index_exactly_once() {
        let n = 10_003;
        let counter: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        par_range(n, 64, |start, end| {
            for i in start..end {
                counter[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        assert!(counter.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn small_ranges_run_inline() {
        let n = 10;
        let counter: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        par_range(n, 64, |start, end| {
            for i in start..end {
                counter[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        assert!(counter.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }
}
