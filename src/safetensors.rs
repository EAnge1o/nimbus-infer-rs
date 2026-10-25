//! A from-scratch `.safetensors` reader: `<u64 LE header length> <JSON header>
//! <raw buffer>`. Supports F32 directly, widens F16 and BF16 to F32 at load
//! time (the engine is float32 end to end, exactly like the Python engine).

use std::collections::HashMap;

use crate::json::{self, Json};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug)]
pub struct SafeTensors {
    pub tensors: HashMap<String, TensorInfo>,
    buffer: Vec<u8>,
}

#[derive(Debug)]
pub struct SafeTensorError(pub String);

impl std::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "safetensors: {}", self.0)
    }
}

impl std::error::Error for SafeTensorError {}

impl SafeTensors {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, SafeTensorError> {
        if bytes.len() < 8 {
            return Err(SafeTensorError("file too small for a header length".into()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| SafeTensorError("header length overflow".into()))?;
        if bytes.len() < header_end {
            return Err(SafeTensorError("header length exceeds file size".into()));
        }
        let header_text = std::str::from_utf8(&bytes[8..header_end])
            .map_err(|_| SafeTensorError("header is not UTF-8".into()))?;
        let header = json::parse(header_text)
            .map_err(|e| SafeTensorError(format!("header JSON: {e}")))?;

        let mut tensors = HashMap::new();
        if let Json::Obj(entries) = header {
            for (name, info) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_str = info
                    .get("dtype")
                    .and_then(Json::as_str)
                    .ok_or_else(|| SafeTensorError(format!("{name}: missing dtype")))?;
                let dtype = match dtype_str {
                    "F32" => DType::F32,
                    "F16" => DType::F16,
                    "BF16" => DType::Bf16,
                    other => {
                        return Err(SafeTensorError(format!(
                            "{name}: unsupported dtype {other} (engine reads F32/F16/BF16)"
                        )))
                    }
                };
                let shape = info
                    .get("shape")
                    .and_then(Json::as_arr)
                    .ok_or_else(|| SafeTensorError(format!("{name}: missing shape")))?
                    .iter()
                    .map(|v| v.as_usize().ok_or_else(|| SafeTensorError(format!("{name}: bad shape"))))
                    .collect::<Result<Vec<usize>, _>>()?;
                let offsets = info
                    .get("data_offsets")
                    .and_then(Json::as_arr)
                    .ok_or_else(|| SafeTensorError(format!("{name}: missing data_offsets")))?;
                if offsets.len() != 2 {
                    return Err(SafeTensorError(format!("{name}: bad data_offsets")));
                }
                let start = offsets[0].as_usize().unwrap();
                let end = offsets[1].as_usize().unwrap();
                tensors.insert(name, TensorInfo { dtype, shape, start, end });
            }
        }
        Ok(SafeTensors { tensors, buffer: bytes[header_end..].to_vec() })
    }

    pub fn open(path: &std::path::Path) -> Result<Self, SafeTensorError> {
        let bytes = std::fs::read(path)
            .map_err(|e| SafeTensorError(format!("reading {}: {e}", path.display())))?;
        Self::from_bytes(bytes)
    }

    pub fn names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        names
    }

    /// Widen any supported dtype to f32 and return a copy.
    pub fn take_f32(&self, name: &str) -> Result<Vec<f32>, SafeTensorError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafeTensorError(format!("missing tensor '{name}'")))?;
        let raw = &self.buffer[info.start..info.end];
        let expected: usize = info.shape.iter().product();
        match info.dtype {
            DType::F32 => {
                if raw.len() != expected * 4 {
                    return Err(SafeTensorError(format!("{name}: size mismatch")));
                }
                Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            DType::F16 => {
                if raw.len() != expected * 2 {
                    return Err(SafeTensorError(format!("{name}: size mismatch")));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect())
            }
            DType::Bf16 => {
                if raw.len() != expected * 2 {
                    return Err(SafeTensorError(format!("{name}: size mismatch")));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                    .collect())
            }
        }
    }
}

/// IEEE half precision -> f32 (handles denormals, inf, nan).
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) as u32) << 31;
    let exp = ((half >> 10) & 0x1F) as u32;
    let frac = (half & 0x3FF) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Denormal: normalize.
                let mut e: i32 = -1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3FF;
                sign | (((127 - 15 + e) as u32) << 23) | (f << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (frac << 13), // inf / nan
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u64(mut v: Vec<u8>, x: u64) -> Vec<u8> {
        v.extend_from_slice(&x.to_le_bytes());
        v
    }

    #[test]
    fn reads_a_handcrafted_f32_file() {
        let header = br#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
        let mut file = write_u64(Vec::new(), header.len() as u64);
        file.extend_from_slice(header);
        for i in 0..6u32 {
            file.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let st = SafeTensors::from_bytes(file).unwrap();
        let w = st.take_f32("w").unwrap();
        assert_eq!(w, (0..6).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn widens_bf16() {
        let header = br#"{"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}"#;
        let mut file = write_u64(Vec::new(), header.len() as u64);
        file.extend_from_slice(header);
        // bf16 1.0 / 2.0 = the top half of the f32 bit pattern.
        let one = (1.0f32.to_bits() >> 16) as u16;
        let two = (2.0f32.to_bits() >> 16) as u16;
        file.extend_from_slice(&one.to_le_bytes());
        file.extend_from_slice(&two.to_le_bytes());
        let st = SafeTensors::from_bytes(file).unwrap();
        let w = st.take_f32("w").unwrap();
        assert_eq!(w, vec![1.0f32, 2.0]);
    }

    #[test]
    fn f16_conversion_edge_cases() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }
}
