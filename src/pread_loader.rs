//! Expert weight loader using pread() for direct I/O instead of mmap.
//!
//! Parses safetensors headers at startup to build a tensor index mapping
//! tensor names to (file_descriptor, byte_offset, byte_length, dtype, shape).
//! On demand, reads exact byte ranges with pread() and creates Metal tensors
//! via Tensor::from_raw_buffer(). Sets F_NOCACHE on file descriptors to bypass
//! the OS page cache, avoiding cache-on-cache interference with Metal buffers.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Mutex;

/// Metadata for a single tensor within a safetensors file.
#[derive(Debug, Clone)]
struct TensorMeta {
    /// Index into the `files` vec.
    file_idx: usize,
    /// Absolute byte offset from start of file to tensor data.
    data_offset: usize,
    /// Byte length of tensor data.
    data_len: usize,
    /// Tensor dtype (safetensors dtype mapped to candle DType).
    dtype: DType,
    /// Tensor shape.
    shape: Vec<usize>,
}

/// Maps safetensors dtype to candle DType.
fn st_dtype_to_candle(dtype: safetensors::Dtype) -> std::result::Result<DType, String> {
    match dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F64 => Ok(DType::F64),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        other => Err(format!("unsupported safetensors dtype: {:?}", other)),
    }
}

/// Holds open file descriptors and a tensor name → metadata index.
/// Designed for use as a `SimpleBackend` for `VarBuilder`.
pub struct PreadTensorLoader {
    files: Vec<File>,
    tensor_index: HashMap<String, TensorMeta>,
    /// Reusable read buffers keyed by byte size. Avoids repeated allocation
    /// and page faulting for the temporary Vec<u8> used by pread.
    read_buffers: Mutex<HashMap<usize, Vec<u8>>>,
}

impl PreadTensorLoader {
    /// Parse safetensors headers from multiple files and build the tensor index.
    /// Sets F_NOCACHE on each file descriptor to bypass OS page cache.
    pub fn new<P: AsRef<Path>>(paths: &[P]) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensor_index = HashMap::new();

        for (file_idx, path) in paths.iter().enumerate() {
            let file = File::open(path.as_ref())?;

            // Set F_NOCACHE to bypass OS page cache on macOS.
            #[cfg(target_os = "macos")]
            {
                let fd = file.as_raw_fd();
                unsafe {
                    libc::fcntl(fd, libc::F_NOCACHE, 1);
                }
            }

            // Read the 8-byte header length.
            let mut header_len_buf = [0u8; 8];
            let n = pread_exact(file.as_raw_fd(), &mut header_len_buf, 0)?;
            if n != 8 {
                return Err(format!("failed to read header length from {:?}", path.as_ref()).into());
            }
            let header_len = u64::from_le_bytes(header_len_buf) as usize;

            // Read the JSON header.
            let mut header_buf = vec![0u8; header_len];
            pread_exact(file.as_raw_fd(), &mut header_buf, 8)?;

            // The data section starts after 8 bytes + header_len.
            let data_start = 8 + header_len;

            // Parse the header to extract tensor metadata.
            // We use safetensors' deserialize on a minimal buffer: header + 0 bytes of data.
            // Instead, parse the JSON directly for offset info.
            let header_json: serde_json::Value = serde_json::from_slice(&header_buf)?;

            if let serde_json::Value::Object(map) = header_json {
                for (name, info) in map {
                    // Skip __metadata__ key
                    if name == "__metadata__" {
                        continue;
                    }
                    let obj = info.as_object().ok_or_else(|| {
                        format!("expected object for tensor '{}' in {:?}", name, path.as_ref())
                    })?;

                    let dtype_str = obj
                        .get("dtype")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| format!("missing dtype for '{}'", name))?;
                    let st_dtype = match dtype_str {
                        "F16" => safetensors::Dtype::F16,
                        "BF16" => safetensors::Dtype::BF16,
                        "F32" => safetensors::Dtype::F32,
                        "F64" => safetensors::Dtype::F64,
                        "U8" => safetensors::Dtype::U8,
                        "U32" => safetensors::Dtype::U32,
                        "I64" => safetensors::Dtype::I64,
                        other => return Err(format!("unsupported dtype '{}' for '{}'", other, name).into()),
                    };
                    let dtype = st_dtype_to_candle(st_dtype)
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

                    let shape: Vec<usize> = obj
                        .get("shape")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| format!("missing shape for '{}'", name))?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect();

                    let offsets = obj
                        .get("data_offsets")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| format!("missing data_offsets for '{}'", name))?;
                    let start = offsets[0].as_u64().unwrap_or(0) as usize;
                    let end = offsets[1].as_u64().unwrap_or(0) as usize;

                    tensor_index.insert(
                        name,
                        TensorMeta {
                            file_idx,
                            data_offset: data_start + start,
                            data_len: end - start,
                            dtype,
                            shape,
                        },
                    );
                }
            }

            files.push(file);
        }

        Ok(Self {
            files,
            tensor_index,
            read_buffers: Mutex::new(HashMap::new()),
        })
    }

    /// Load a single tensor by name onto the given device.
    fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let meta = self.tensor_index.get(name).ok_or_else(|| {
            candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            }
            .bt()
        })?;

        let fd = self.files[meta.file_idx].as_raw_fd();
        let size = meta.data_len;

        // Take the reusable buffer from the pool (or allocate + fault in a new one).
        let t0 = std::time::Instant::now();
        let mut buf = self
            .read_buffers
            .lock().unwrap()
            .remove(&size)
            .unwrap_or_else(|| {
                let mut b = vec![0u8; size];
                // Touch every page to force the OS to map physical pages now,
                // so pread doesn't pay for page faults later.
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
                for offset in (0..size).step_by(page_size) {
                    b[offset] = 0;
                }
                b
            });
        let t1 = std::time::Instant::now();

        pread_exact(fd, &mut buf, meta.data_offset as i64).map_err(|e| {
            candle_core::Error::Msg(format!("pread failed for '{}': {}", name, e))
        })?;
        let t2 = std::time::Instant::now();

        let tensor = Tensor::from_raw_buffer(&buf, meta.dtype, &meta.shape, device)?;
        let t3 = std::time::Instant::now();

        // Return buffer to the pool for reuse.
        self.read_buffers.lock().unwrap().insert(size, buf);

        println!(
            "  load_tensor({}): buf={:.1}ms pread={:.1}ms from_raw_buffer={:.1}ms total={:.1}ms",
            name.rsplit('.').nth(1).unwrap_or(name),
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
            (t3 - t0).as_secs_f64() * 1000.0,
        );
        Ok(tensor)
    }

    /// Create a VarBuilder backed by this pread loader with a prefix prepended
    /// to all tensor name lookups. For example, with prefix
    /// `"model.layers.0.block_sparse_moe.experts.0"`, a VarBuilder `.pp("w1")`
    /// lookup for `"w1.weight"` resolves to the full tensor name
    /// `"model.layers.0.block_sparse_moe.experts.0.w1.weight"`.
    pub fn var_builder(&self, dtype: DType, device: &Device, prefix: &str) -> VarBuilder<'_> {
        let backend = PreadBackend {
            loader: self,
            prefix: prefix.to_string(),
        };
        VarBuilder::from_backend(Box::new(backend), dtype, device.clone())
    }
}

/// Wrapper to implement SimpleBackend, prepending a fixed prefix to tensor names.
struct PreadBackend<'a> {
    loader: &'a PreadTensorLoader,
    prefix: String,
}

impl PreadBackend<'_> {
    fn full_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}

impl candle_nn::var_builder::SimpleBackend for PreadBackend<'_> {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let full = self.full_name(name);
        let tensor = self.loader.load_tensor(&full, dev)?;
        if tensor.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!(
                    "shape mismatch for '{}': expected {:?}, got {:?}",
                    full,
                    s,
                    tensor.shape()
                ),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        tensor.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let full = self.full_name(name);
        self.loader.load_tensor(&full, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let full = self.full_name(name);
        self.loader.tensor_index.contains_key(&full)
    }
}

// Safety: PreadBackend holds a shared reference to PreadTensorLoader (File handles
// + HashMap, both Send+Sync) and a String. pread() is thread-safe as it doesn't
// modify the file descriptor's seek offset.
unsafe impl Send for PreadBackend<'_> {}
unsafe impl Sync for PreadBackend<'_> {}

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
                format!("pread: unexpected EOF at offset {} (read {}/{})", offset, filled, total),
            ));
        }
        filled += n as usize;
        offset += n as i64;
    }
    Ok(filled)
}
