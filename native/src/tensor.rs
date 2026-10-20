//! Read-only safetensors access over mmap'd shards.
//!
//! We deliberately do not load a whole checkpoint: the 27B merged weights are
//! ~51 GiB. Tensors are copied to the GPU one at a time and embedding rows are
//! read straight out of the mapped file.
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use half::bf16;
use memmap2::Mmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
}

impl DType {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" | "BOOL" => DType::U8,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub shard: usize,
    /// absolute byte offsets in the mapped file
    pub start: usize,
    pub end: usize,
}

struct Shard {
    _path: PathBuf,
    mmap: Mmap,
}

pub struct Store {
    shards: Vec<Shard>,
    map: HashMap<String, TensorInfo>,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        let index = dir.join("model.safetensors.index.json");
        let single = dir.join("model.safetensors");
        if single.exists() && !index.exists() {
            files.push(single);
        } else if index.exists() {
            let raw = std::fs::read_to_string(&index)?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            let wm = v
                .get("weight_map")
                .and_then(|x| x.as_object())
                .context("index.json has no weight_map")?;
            let mut seen = std::collections::BTreeSet::new();
            for name in wm.values() {
                if let Some(s) = name.as_str() {
                    seen.insert(s.to_string());
                }
            }
            for f in seen {
                files.push(dir.join(f));
            }
        } else {
            for e in std::fs::read_dir(dir)? {
                let p = e?.path();
                if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
                    files.push(p);
                }
            }
            files.sort();
        }
        if files.is_empty() {
            bail!("no safetensors files in {}", dir.display());
        }

        let mut shards = Vec::new();
        let mut map = HashMap::new();
        for (si, path) in files.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };
            let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])
                .with_context(|| format!("parsing header of {}", path.display()))?;
            let data_start = 8 + hlen;
            let obj = header.as_object().context("header is not an object")?;
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = DType::parse(info["dtype"].as_str().context("dtype")?)?;
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(0) as usize)
                    .collect();
                let off = info["data_offsets"].as_array().context("data_offsets")?;
                let start = data_start + off[0].as_u64().unwrap() as usize;
                let end = data_start + off[1].as_u64().unwrap() as usize;
                map.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        shape,
                        shard: si,
                        start,
                        end,
                    },
                );
            }
            shards.push(Shard {
                _path: path.clone(),
                mmap,
            });
        }
        Ok(Self { shards, map })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.map.get(name)
    }

    /// Map a single `.safetensors` file (for example a LoRA adapter).
    pub fn open_file(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])
            .with_context(|| format!("parsing header of {}", path.display()))?;
        let data_start = 8 + hlen;
        let obj = header.as_object().context("header is not an object")?;
        let mut map = HashMap::new();
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = DType::parse(info["dtype"].as_str().context("dtype")?)?;
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = info["data_offsets"].as_array().context("data_offsets")?;
            let start = data_start + off[0].as_u64().unwrap() as usize;
            let end = data_start + off[1].as_u64().unwrap() as usize;
            map.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    shard: 0,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            shards: vec![Shard {
                _path: path.to_path_buf(),
                mmap,
            }],
            map,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Copy a BF16 tensor to a host Vec.
    pub fn bf16(&self, name: &str) -> Result<Vec<bf16>> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        if info.dtype != DType::BF16 {
            bail!("{name}: expected BF16, got {:?}", info.dtype);
        }
        let bytes = &self.shards[info.shard].mmap[info.start..info.end];
        Ok(bf16_slice(bytes).to_vec())
    }

    /// Copy a scalar tensor to host f32, accepting F32 or BF16 storage.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        let bytes = &self.shards[info.shard].mmap[info.start..info.end];
        match info.dtype {
            DType::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            DType::BF16 => Ok(bf16_slice(bytes).iter().map(|v| v.to_f32()).collect()),
            other => bail!("{name}: expected F32/BF16, got {other:?}"),
        }
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        if info.dtype != DType::I32 {
            bail!("{name}: expected I32, got {:?}", info.dtype);
        }
        let bytes = &self.shards[info.shard].mmap[info.start..info.end];
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        if info.dtype != DType::I64 {
            bail!("{name}: expected I64, got {:?}", info.dtype);
        }
        let bytes = &self.shards[info.shard].mmap[info.start..info.end];
        Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Copy `cols` BF16 elements of `row` into `out`.
    pub fn bf16_row(&self, name: &str, row: usize, out: &mut [bf16]) -> Result<()> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        if info.dtype != DType::BF16 || info.shape.len() != 2 {
            bail!("{name}: expected 2-D BF16");
        }
        let cols = info.shape[1];
        if out.len() != cols || row >= info.shape[0] {
            bail!("{name}: bad row/cols");
        }
        let base = info.start + row * cols * 2;
        let bytes = &self.shards[info.shard].mmap[base..base + cols * 2];
        out.copy_from_slice(bf16_slice(bytes));
        Ok(())
    }

    /// Borrow a whole BF16 tensor from the mmap (no copy).
    pub fn bf16_slice(&self, name: &str) -> Result<&[bf16]> {
        let info = self
            .map
            .get(name)
            .with_context(|| format!("tensor {name} not found"))?;
        if info.dtype != DType::BF16 {
            bail!("{name}: expected BF16, got {:?}", info.dtype);
        }
        let bytes = &self.shards[info.shard].mmap[info.start..info.end];
        if bytes.len() % 2 != 0 {
            bail!("{name}: odd BF16 byte length");
        }
        let ptr = bytes.as_ptr();
        if ptr as usize % 2 != 0 {
            bail!("{name}: unaligned BF16 storage");
        }
        // SAFETY: BF16 is two bytes, the range is aligned, and the mmap lives
        // as long as the Store.
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const bf16, bytes.len() / 2) })
    }

    /// Build a name-to-shape summary, used for startup diagnostics.
    pub fn tensor_names(&self) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.map.iter()
    }
}

fn bf16_slice(bytes: &[u8]) -> &[bf16] {
    debug_assert!(bytes.len() % 2 == 0);
    // bf16 is repr(transparent) over u16; safetensors payloads are 2-byte aligned
    // relative to a page-aligned mmap.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2) }
}
