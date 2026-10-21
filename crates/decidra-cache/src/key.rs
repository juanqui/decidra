//! Cache keys: length-prefixed hashing, backbone/program hashes, numeric config,
//! and the tier keys.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type Hash32 = [u8; 32];

/// Domain-separated, length-prefixed hash. Every field is framed, so
/// `("ab","c")` and `("a","bc")` cannot collide.
pub fn hash_fields(domain: &[u8], fields: &[&[u8]]) -> Hash32 {
    let mut h = Sha256::new();
    h.update((domain.len() as u64).to_le_bytes());
    h.update(domain);
    h.update((fields.len() as u64).to_le_bytes());
    for field in fields {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field);
    }
    h.finalize().into()
}

pub fn hex(hash: &Hash32) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecVariant {
    /// One-item batch through the single-sequence path.
    Single,
    /// Batched path; `m` is the total row count class that selects GEMM/GDN kernels.
    Batched { m: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProgramHash(pub Hash32);

/// Inputs to the backbone hash. All fields are part of the function.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackboneInputs {
    pub weights_manifest_sha256: Hash32,
    pub adapter_sha256: Option<Hash32>,
    pub tokenizer_sha256: Hash32,
    pub mask_family: u8,
    pub numeric_config_hash: Hash32,
}

impl BackboneInputs {
    pub fn h(&self) -> Hash32 {
        let adapter = match self.adapter_sha256 {
            Some(h) => h.to_vec(),
            None => b"none".to_vec(),
        };
        let family = [self.mask_family];
        hash_fields(
            b"decidra-backbone-v1",
            &[
                &self.weights_manifest_sha256,
                &adapter,
                &self.tokenizer_sha256,
                &family,
                &self.numeric_config_hash,
            ],
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReadoutInputs {
    pub readout_sha256: Hash32,
    pub readout_dtype: u8,
    pub index_sha256: Hash32,
}

impl ReadoutInputs {
    pub fn program(&self, backbone: &Hash32) -> ProgramHash {
        ProgramHash(hash_fields(
            b"decidra-program-v1",
            &[
                backbone,
                &self.readout_sha256,
                &[self.readout_dtype],
                &self.index_sha256,
            ],
        ))
    }
}

/// T0 exact-decision key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub program_hash: ProgramHash,
    pub exec_variant: ExecVariant,
    pub tokens: Vec<u32>,
    pub head_ids: Vec<u32>,
}

/// T1 site-hidden key: independent of head ids and temperature.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteHiddenKey {
    pub backbone_hash: Hash32,
    pub exec_variant: ExecVariant,
    pub tokens: Vec<u32>,
    pub image_ids: Vec<Hash32>,
    pub pos_ids: Vec<u32>,
    pub sites: Vec<u32>,
}

/// Knobs that change logits. Frozen after preload; every field is hashed.
#[derive(Debug, Clone)]
pub struct NumericConfig {
    marlin: bool,
    marlin_fp8: bool,
    marlin_int8: bool,
    int8_clip: f32,
    int8_group_scale: bool,
    int8_group_exclude: usize,
    int8_exclude: bool,
    int8_all_layers: bool,
    fused_w4a16: bool,
    fused_min_n: usize,
    fused_max_m: usize,
    fused_tile128: bool,
    warp_attention: bool,
    flashinfer_enabled: bool,
    flashinfer_min_seq: usize,
    gdn_shared: bool,
    gdn_split: bool,
    gdn_split_min_seq: usize,
    gdn_chunks: usize,
    gdn_chunked: bool,
    gdn_chunked_tc: bool,
    gdn_chunked_min_seq: usize,
    conv_tiled: bool,
    gpu_embed: bool,
    marlin_chunk: u64,
    int8_group_uniform: bool,
    frozen: bool,
}

impl Default for NumericConfig {
    fn default() -> Self {
        Self {
            marlin: true,
            marlin_fp8: false,
            marlin_int8: true,
            int8_clip: 1.0,
            int8_group_scale: true,
            int8_group_exclude: 1,
            int8_exclude: false,
            int8_all_layers: true,
            fused_w4a16: false,
            fused_min_n: 0,
            fused_max_m: 0,
            fused_tile128: false,
            warp_attention: false,
            flashinfer_enabled: true,
            flashinfer_min_seq: 1024,
            gdn_shared: true,
            gdn_split: true,
            gdn_split_min_seq: 0,
            gdn_chunks: 0,
            gdn_chunked: true,
            gdn_chunked_tc: true,
            gdn_chunked_min_seq: 64,
            conv_tiled: false,
            gpu_embed: false,
            marlin_chunk: 0,
            int8_group_uniform: false,
            frozen: false,
        }
    }
}

impl NumericConfig {
    pub fn freeze(&mut self) {
        self.frozen = true;
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    fn ensure_mutable(&self) -> Result<(), &'static str> {
        if self.frozen {
            Err("numeric config is frozen after preload")
        } else {
            Ok(())
        }
    }

    pub fn set_marlin(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.marlin = v;
        Ok(())
    }
    pub fn set_marlin_fp8(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.marlin_fp8 = v;
        Ok(())
    }
    pub fn set_marlin_int8(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.marlin_int8 = v;
        Ok(())
    }
    pub fn set_int8_clip(&mut self, v: f32) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_clip = v;
        Ok(())
    }
    pub fn set_int8_group_scale(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_group_scale = v;
        Ok(())
    }
    pub fn set_int8_group_exclude(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_group_exclude = v;
        Ok(())
    }
    pub fn set_int8_exclude(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_exclude = v;
        Ok(())
    }
    pub fn set_int8_all_layers(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_all_layers = v;
        Ok(())
    }
    pub fn set_fused_w4a16(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.fused_w4a16 = v;
        Ok(())
    }
    pub fn set_fused_min_n(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.fused_min_n = v;
        Ok(())
    }
    pub fn set_fused_max_m(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.fused_max_m = v;
        Ok(())
    }
    pub fn set_fused_tile128(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.fused_tile128 = v;
        Ok(())
    }
    pub fn set_warp_attention(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.warp_attention = v;
        Ok(())
    }
    pub fn set_flashinfer_enabled(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.flashinfer_enabled = v;
        Ok(())
    }
    pub fn set_flashinfer_min_seq(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.flashinfer_min_seq = v;
        Ok(())
    }
    pub fn set_gdn_shared(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_shared = v;
        Ok(())
    }
    pub fn set_gdn_split(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_split = v;
        Ok(())
    }
    pub fn set_gdn_split_min_seq(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_split_min_seq = v;
        Ok(())
    }
    pub fn set_gdn_chunks(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_chunks = v;
        Ok(())
    }
    pub fn set_gdn_chunked(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_chunked = v;
        Ok(())
    }
    pub fn set_gdn_chunked_tc(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_chunked_tc = v;
        Ok(())
    }
    pub fn set_gdn_chunked_min_seq(&mut self, v: usize) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gdn_chunked_min_seq = v;
        Ok(())
    }
    pub fn set_conv_tiled(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.conv_tiled = v;
        Ok(())
    }
    pub fn set_gpu_embed(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.gpu_embed = v;
        Ok(())
    }
    pub fn set_marlin_chunk(&mut self, v: u64) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.marlin_chunk = v;
        Ok(())
    }
    pub fn set_int8_group_uniform(&mut self, v: bool) -> Result<(), &'static str> {
        self.ensure_mutable()?;
        self.int8_group_uniform = v;
        Ok(())
    }

    /// Every numeric field, in a stable order. The test asserts the count and
    /// that each one changes the hash.
    pub fn field_bytes(&self) -> Vec<Vec<u8>> {
        vec![
            vec![self.marlin as u8],
            vec![self.marlin_fp8 as u8],
            vec![self.marlin_int8 as u8],
            self.int8_clip.to_bits().to_le_bytes().to_vec(),
            vec![self.int8_group_scale as u8],
            self.int8_group_exclude.to_le_bytes().to_vec(),
            vec![self.int8_exclude as u8],
            vec![self.int8_all_layers as u8],
            vec![self.fused_w4a16 as u8],
            self.fused_min_n.to_le_bytes().to_vec(),
            self.fused_max_m.to_le_bytes().to_vec(),
            vec![self.fused_tile128 as u8],
            vec![self.warp_attention as u8],
            vec![self.flashinfer_enabled as u8],
            self.flashinfer_min_seq.to_le_bytes().to_vec(),
            vec![self.gdn_shared as u8],
            vec![self.gdn_split as u8],
            self.gdn_split_min_seq.to_le_bytes().to_vec(),
            self.gdn_chunks.to_le_bytes().to_vec(),
            vec![self.gdn_chunked as u8],
            vec![self.gdn_chunked_tc as u8],
            self.gdn_chunked_min_seq.to_le_bytes().to_vec(),
            vec![self.conv_tiled as u8],
            vec![self.gpu_embed as u8],
            self.marlin_chunk.to_le_bytes().to_vec(),
            vec![self.int8_group_uniform as u8],
        ]
    }

    pub fn hash(&self) -> Hash32 {
        let fields = self.field_bytes();
        let refs: Vec<&[u8]> = fields.iter().map(|f| f.as_slice()).collect();
        hash_fields(b"decidra-numeric-v1", &refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_key_lenprefix() {
        let a = hash_fields(b"x", &[b"ab", b"c"]);
        let b = hash_fields(b"x", &[b"a", b"bc"]);
        assert_ne!(a, b);
        let c = hash_fields(b"x", &[b"ab", b"c"]);
        assert_eq!(a, c);
        let d = hash_fields(b"y", &[b"ab", b"c"]);
        assert_ne!(a, d);
    }

    #[test]
    fn t_key_numeric() {
        let base = NumericConfig::default();
        let n = base.field_bytes().len();
        assert_eq!(
            n, 26,
            "NumericConfig field count changed; add it to field_bytes()/hash()"
        );
        let h0 = base.hash();
        let mut cfg = base.clone();
        cfg.set_marlin(false).unwrap();
        assert_ne!(cfg.hash(), h0);
        let mut cfg = base.clone();
        cfg.set_int8_clip(0.99).unwrap();
        assert_ne!(cfg.hash(), h0);
        let mut cfg = base.clone();
        cfg.set_flashinfer_min_seq(0).unwrap();
        assert_ne!(cfg.hash(), h0);
        let mut cfg = base.clone();
        cfg.set_marlin_chunk(7).unwrap();
        assert_ne!(cfg.hash(), h0);
        let mut cfg = base.clone();
        cfg.set_int8_group_uniform(true).unwrap();
        assert_ne!(cfg.hash(), h0);
    }

    #[test]
    fn t_key_frozen() {
        let mut cfg = NumericConfig::default();
        cfg.freeze();
        assert!(cfg.set_marlin(false).is_err());
        assert!(cfg.set_gdn_shared(false).is_err());
        assert!(cfg.set_int8_clip(0.9).is_err());
    }

    #[test]
    fn t_key_mask() {
        let numeric = NumericConfig::default().hash();
        let base = BackboneInputs {
            weights_manifest_sha256: [1; 32],
            adapter_sha256: None,
            tokenizer_sha256: [2; 32],
            mask_family: 0,
            numeric_config_hash: numeric,
        };
        let mut other = base.clone();
        other.mask_family = 1;
        assert_ne!(base.h(), other.h());
        let mut other = base.clone();
        other.adapter_sha256 = Some([9; 32]);
        assert_ne!(base.h(), other.h());
        let mut other = base.clone();
        other.numeric_config_hash = [7; 32];
        assert_ne!(base.h(), other.h());
    }

    #[test]
    fn t_key_variant_pos_split() {
        let key = CacheKey {
            program_hash: ProgramHash([3; 32]),
            exec_variant: ExecVariant::Single,
            tokens: vec![1, 2, 3],
            head_ids: vec![0, 1],
        };
        let mut other = key.clone();
        other.exec_variant = ExecVariant::Batched { m: 4 };
        assert_ne!(key, other);
        let site = SiteHiddenKey {
            backbone_hash: [1; 32],
            exec_variant: ExecVariant::Single,
            tokens: vec![1, 2, 3],
            image_ids: vec![],
            pos_ids: vec![0, 1, 2],
            sites: vec![2],
        };
        let mut other = site.clone();
        other.pos_ids = vec![3, 4, 5];
        assert_ne!(site, other);
        let mut other = site.clone();
        other.sites = vec![1];
        assert_ne!(site, other);
        let mut other = site.clone();
        other.image_ids = vec![[4; 32]];
        assert_ne!(site, other);
    }
}
