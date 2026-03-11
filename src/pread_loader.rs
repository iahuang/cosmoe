//! Expert weight loader using pread() for direct I/O from GGUF files.
//!
//! Parses the GGUF header at startup to build a tensor index mapping
//! tensor names to (byte_offset, byte_length, ggml_dtype, shape).
//! On demand, reads exact byte ranges with pread() and creates QTensors
//! via qtensor_from_ggml(). Sets F_NOCACHE on the file descriptor to bypass
//! the OS page cache, avoiding cache-on-cache interference with Metal buffers.
//!
//! This loader expects per-expert tensors (unfused):
//!   blk.{layer}.ffn_gate.{expert}.weight  (w1)
//!   blk.{layer}.ffn_down.{expert}.weight  (w2)
//!   blk.{layer}.ffn_up.{expert}.weight    (w3)

use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Holds an open file descriptor and the parsed GGUF tensor index.
/// Loads tensors on demand via pread() into QTensors.
pub struct PreadTensorLoader {
    file: File,
    tensor_infos: HashMap<String, gguf_file::TensorInfo>,
    tensor_data_offset: u64,
    /// GGUF metadata (model hyperparameters, etc.)
    pub metadata: HashMap<String, gguf_file::Value>,
    /// Reusable read buffers keyed by byte size. Avoids repeated allocation
    /// and page faulting for the temporary Vec<u8> used by pread.
    read_buffers: Mutex<HashMap<usize, Vec<u8>>>,
    /// Cumulative bytes read via pread (atomic for thread-safe access from I/O thread).
    pub bytes_read: AtomicU64,
    /// Cumulative time spent in pread syscalls (nanoseconds).
    pub pread_nanos: AtomicU64,
    /// Cumulative time spent creating QTensors from raw data (nanoseconds).
    pub qtensor_nanos: AtomicU64,
    /// Number of tensors loaded.
    pub tensors_loaded: AtomicU64,
}

impl PreadTensorLoader {
    /// Parse the GGUF header and build the tensor index.
    /// Sets F_NOCACHE on the file descriptor to bypass OS page cache.
    pub fn new<P: AsRef<Path>>(path: P) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path.as_ref())?;

        // Set F_NOCACHE to bypass OS page cache on macOS.
        #[cfg(target_os = "macos")]
        {
            let fd = file.as_raw_fd();
            unsafe {
                libc::fcntl(fd, libc::F_NOCACHE, 1);
            }
        }

        // Parse the GGUF header (metadata + tensor infos).
        let mut reader = BufReader::new(&file);
        let content = gguf_file::Content::read(&mut reader)?;

        Ok(Self {
            file,
            tensor_infos: content.tensor_infos,
            tensor_data_offset: content.tensor_data_offset,
            metadata: content.metadata,
            read_buffers: Mutex::new(HashMap::new()),
            bytes_read: AtomicU64::new(0),
            pread_nanos: AtomicU64::new(0),
            qtensor_nanos: AtomicU64::new(0),
            tensors_loaded: AtomicU64::new(0),
        })
    }

    /// Reset I/O stats (call after startup loading to measure only expert loading).
    pub fn reset_stats(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
        self.pread_nanos.store(0, Ordering::Relaxed);
        self.qtensor_nanos.store(0, Ordering::Relaxed);
        self.tensors_loaded.store(0, Ordering::Relaxed);
    }

    /// Print I/O throughput summary.
    pub fn print_io_stats(&self) {
        let bytes = self.bytes_read.load(Ordering::Relaxed);
        let pread_ns = self.pread_nanos.load(Ordering::Relaxed);
        let qtensor_ns = self.qtensor_nanos.load(Ordering::Relaxed);
        let count = self.tensors_loaded.load(Ordering::Relaxed);

        if count == 0 {
            return;
        }

        let pread_secs = pread_ns as f64 / 1e9;
        let qtensor_secs = qtensor_ns as f64 / 1e9;
        let bytes_f = bytes as f64;

        let throughput_gb_s = if pread_secs > 0.0 {
            (bytes_f / (1024.0 * 1024.0 * 1024.0)) / pread_secs
        } else {
            0.0
        };

        println!(
            "I/O Stats:   {} tensors, {:.1} MB read, pread {:.1}ms ({:.2} GB/s), qtensor {:.1}ms",
            count,
            bytes_f / (1024.0 * 1024.0),
            pread_secs * 1000.0,
            throughput_gb_s,
            qtensor_secs * 1000.0,
        );
    }

    /// Load a single tensor by name onto the given device as a QTensor.
    pub fn load_qtensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let info = self.tensor_infos.get(name).ok_or_else(|| {
            candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            }
            .bt()
        })?;

        let elem_count: usize = info.shape.elem_count();
        let block_size = info.ggml_dtype.block_size();
        let size = elem_count / block_size * info.ggml_dtype.type_size();
        let absolute_offset = self.tensor_data_offset + info.offset;
        let fd = self.file.as_raw_fd();

        // Take the reusable buffer from the pool (or allocate + fault in a new one).
        let mut buf = self
            .read_buffers
            .lock()
            .unwrap()
            .remove(&size)
            .unwrap_or_else(|| {
                let mut b = vec![0u8; size];
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
                for offset in (0..size).step_by(page_size) {
                    b[offset] = 0;
                }
                b
            });

        let t_pread = std::time::Instant::now();
        pread_exact(fd, &mut buf, absolute_offset as i64).map_err(|e| {
            candle_core::Error::Msg(format!("pread failed for '{}': {}", name, e))
        })?;
        let pread_elapsed = t_pread.elapsed();

        let t_qtensor = std::time::Instant::now();
        let qtensor = qtensor_from_ggml(
            info.ggml_dtype,
            &buf,
            info.shape.dims().to_vec(),
            device,
        )?;
        let qtensor_elapsed = t_qtensor.elapsed();

        // Return buffer to the pool for reuse.
        self.read_buffers.lock().unwrap().insert(size, buf);

        // Accumulate stats.
        self.bytes_read.fetch_add(size as u64, Ordering::Relaxed);
        self.pread_nanos.fetch_add(pread_elapsed.as_nanos() as u64, Ordering::Relaxed);
        self.qtensor_nanos.fetch_add(qtensor_elapsed.as_nanos() as u64, Ordering::Relaxed);
        self.tensors_loaded.fetch_add(1, Ordering::Relaxed);

        Ok(qtensor)
    }
}

/// pread() wrapper that reads exactly `buf.len()` bytes, retrying on partial reads.
fn pread_exact(fd: i32, buf: &mut [u8], mut offset: i64) -> std::io::Result<usize> {
    let total = buf.len();
    let mut filled = 0;
    while filled < total {
        let n = unsafe {
            libc::pread(
                fd,
                buf[filled..].as_mut_ptr() as *mut libc::c_void,
                total - filled,
                offset,
            )
        };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "pread: unexpected EOF at offset {} (read {}/{})",
                    offset, filled, total
                ),
            ));
        }
        filled += n as usize;
        offset += n as i64;
    }
    Ok(filled)
}
