use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::{
    backend::{Rocm, rocm::RocmDevice},
    module::{Initializer, Module, Param},
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        attention::{
            MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
        },
        conv::{Conv2d, Conv2dConfig},
    },
    store::ModuleSnapshot as _,
    tensor::{Int, Tensor, activation::sigmoid},
};
use burn_store::pytorch::PytorchStore;
use image::{DynamicImage, imageops::FilterType};
use rusqlite::params;
use tokenizers::Tokenizer;

use crate::{
    db::{Database, path_from_bytes},
    model::{AiReport, EMBEDDING_DIMENSION, LabelPack, list_label_packs, model_key, softmax},
};

type Backend = Rocm<f32, i32, u8>;

const IMAGE_SIZE: usize = 224;
const PATCH_SIZE: usize = 32;
const VISION_WIDTH: usize = 768;
const TEXT_WIDTH: usize = 512;
const CONTEXT_LENGTH: usize = 77;
const VOCAB_SIZE: usize = 49_408;
const EOT_TOKEN: u32 = 49_407;

#[derive(Module, Debug)]
struct ClipModel<B: burn::tensor::backend::Backend> {
    text_model: ClipTextTransformer<B>,
    vision_model: ClipVisionTransformer<B>,
    visual_projection: Linear<B>,
    text_projection: Linear<B>,
}

#[derive(Module, Debug)]
struct ClipTextTransformer<B: burn::tensor::backend::Backend> {
    embeddings: ClipTextEmbeddings<B>,
    encoder: ClipEncoder<B>,
    final_layer_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct ClipVisionTransformer<B: burn::tensor::backend::Backend> {
    embeddings: ClipVisionEmbeddings<B>,
    pre_layrnorm: LayerNorm<B>,
    encoder: ClipEncoder<B>,
    post_layernorm: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct ClipTextEmbeddings<B: burn::tensor::backend::Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
}

#[derive(Module, Debug)]
struct ClipVisionEmbeddings<B: burn::tensor::backend::Backend> {
    class_embedding: Param<Tensor<B, 1>>,
    patch_embedding: Conv2d<B>,
    position_embedding: Embedding<B>,
}

#[derive(Module, Debug)]
struct ClipEncoder<B: burn::tensor::backend::Backend> {
    layers: Vec<ClipEncoderLayer<B>>,
}

#[derive(Module, Debug)]
struct ClipEncoderLayer<B: burn::tensor::backend::Backend> {
    self_attn: MultiHeadAttention<B>,
    layer_norm1: LayerNorm<B>,
    mlp: ClipMlp<B>,
    layer_norm2: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct ClipMlp<B: burn::tensor::backend::Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

struct ClipEngine {
    model: ClipModel<Backend>,
    tokenizer: Tokenizer,
    device: RocmDevice,
}

impl<B: burn::tensor::backend::Backend> ClipModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            text_model: ClipTextTransformer {
                embeddings: ClipTextEmbeddings {
                    token_embedding: EmbeddingConfig::new(VOCAB_SIZE, TEXT_WIDTH).init(device),
                    position_embedding: EmbeddingConfig::new(CONTEXT_LENGTH, TEXT_WIDTH)
                        .init(device),
                },
                encoder: ClipEncoder::new(12, TEXT_WIDTH, 2_048, 8, device),
                final_layer_norm: LayerNormConfig::new(TEXT_WIDTH)
                    .with_epsilon(1e-5)
                    .init(device),
            },
            vision_model: ClipVisionTransformer {
                embeddings: ClipVisionEmbeddings {
                    class_embedding: Initializer::Zeros.init([VISION_WIDTH], device),
                    patch_embedding: Conv2dConfig::new([3, VISION_WIDTH], [PATCH_SIZE, PATCH_SIZE])
                        .with_stride([PATCH_SIZE, PATCH_SIZE])
                        .with_bias(false)
                        .init(device),
                    position_embedding: EmbeddingConfig::new(50, VISION_WIDTH).init(device),
                },
                pre_layrnorm: LayerNormConfig::new(VISION_WIDTH)
                    .with_epsilon(1e-5)
                    .init(device),
                encoder: ClipEncoder::new(12, VISION_WIDTH, 3_072, 12, device),
                post_layernorm: LayerNormConfig::new(VISION_WIDTH)
                    .with_epsilon(1e-5)
                    .init(device),
            },
            visual_projection: LinearConfig::new(VISION_WIDTH, EMBEDDING_DIMENSION)
                .with_bias(false)
                .init(device),
            text_projection: LinearConfig::new(TEXT_WIDTH, EMBEDDING_DIMENSION)
                .with_bias(false)
                .init(device),
        }
    }

    fn encode_image(&self, pixels: Tensor<B, 4>) -> Tensor<B, 2> {
        let hidden = self.vision_model.embeddings.forward(pixels);
        let hidden = self.vision_model.pre_layrnorm.forward(hidden);
        let hidden = self.vision_model.encoder.forward(hidden, None);
        let batch = hidden.dims()[0];
        let pooled = hidden.slice([0..batch, 0..1, 0..VISION_WIDTH]);
        let pooled = pooled.reshape([batch, VISION_WIDTH]);
        let pooled = self.vision_model.post_layernorm.forward(pooled);
        normalize(self.visual_projection.forward(pooled))
    }

    fn encode_text(&self, tokens: Tensor<B, 2, Int>, eot_index: usize) -> Tensor<B, 2> {
        let [batch, _] = tokens.dims();
        let hidden = self.text_model.embeddings.forward(tokens);
        let mask = generate_autoregressive_mask(batch, CONTEXT_LENGTH, &hidden.device());
        let hidden = self.text_model.encoder.forward(hidden, Some(mask));
        let hidden = self.text_model.final_layer_norm.forward(hidden);
        let pooled = hidden
            .slice([0..batch, eot_index..eot_index + 1, 0..TEXT_WIDTH])
            .reshape([batch, TEXT_WIDTH]);
        normalize(self.text_projection.forward(pooled))
    }
}

impl<B: burn::tensor::backend::Backend> ClipTextEmbeddings<B> {
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, sequence] = tokens.dims();
        let token_embeddings = self.token_embedding.forward(tokens);
        let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &token_embeddings.device())
            .reshape([1, sequence])
            .expand([batch, sequence]);
        token_embeddings + self.position_embedding.forward(positions)
    }
}

impl<B: burn::tensor::backend::Backend> ClipVisionEmbeddings<B> {
    fn forward(&self, pixels: Tensor<B, 4>) -> Tensor<B, 3> {
        let patches = self.patch_embedding.forward(pixels);
        let [batch, channels, height, width] = patches.dims();
        let patches = patches
            .reshape([batch, channels, height * width])
            .swap_dims(1, 2);
        let class = self
            .class_embedding
            .val()
            .reshape([1, 1, channels])
            .expand([batch, 1, channels]);
        let embeddings = Tensor::cat(vec![class, patches], 1);
        let sequence = embeddings.dims()[1];
        let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &embeddings.device())
            .reshape([1, sequence])
            .expand([batch, sequence]);
        embeddings + self.position_embedding.forward(positions)
    }
}

impl<B: burn::tensor::backend::Backend> ClipEncoder<B> {
    fn new(
        layer_count: usize,
        width: usize,
        intermediate: usize,
        heads: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| ClipEncoderLayer::new(width, intermediate, heads, device))
                .collect(),
        }
    }

    fn forward(
        &self,
        mut hidden: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            hidden = layer.forward(hidden, mask.clone());
        }
        hidden
    }
}

impl<B: burn::tensor::backend::Backend> ClipEncoderLayer<B> {
    fn new(width: usize, intermediate: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            self_attn: MultiHeadAttentionConfig::new(width, heads)
                .with_dropout(0.0)
                .init(device),
            layer_norm1: LayerNormConfig::new(width).with_epsilon(1e-5).init(device),
            mlp: ClipMlp {
                fc1: LinearConfig::new(width, intermediate).init(device),
                fc2: LinearConfig::new(intermediate, width).init(device),
            },
            layer_norm2: LayerNormConfig::new(width).with_epsilon(1e-5).init(device),
        }
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let normalized = self.layer_norm1.forward(hidden.clone());
        let mut attention = MhaInput::self_attn(normalized);
        if let Some(mask) = mask {
            attention = attention.mask_attn(mask);
        }
        let hidden = hidden + self.self_attn.forward(attention).context;
        let normalized = self.layer_norm2.forward(hidden.clone());
        hidden + self.mlp.forward(normalized)
    }
}

impl<B: burn::tensor::backend::Backend> ClipMlp<B> {
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.fc1.forward(hidden);
        let activated = hidden.clone() * sigmoid(hidden.mul_scalar(1.702));
        self.fc2.forward(activated)
    }
}

impl ClipEngine {
    fn load(directory: &Path) -> Result<Self> {
        let device = RocmDevice::default();
        let mut model = ClipModel::<Backend>::new(&device);
        let mut store = PytorchStore::from_file(directory.join("pytorch_model.bin"))
            .with_key_remapping(r"\.self_attn\.q_proj\.", ".self_attn.query.")
            .with_key_remapping(r"\.self_attn\.k_proj\.", ".self_attn.key.")
            .with_key_remapping(r"\.self_attn\.v_proj\.", ".self_attn.value.")
            .with_key_remapping(r"\.self_attn\.out_proj\.", ".self_attn.output.")
            .allow_partial(false);
        let result = model
            .load_from(&mut store)
            .context("failed to load CLIP weights into Burn")?;
        if !result.errors.is_empty() || !result.missing.is_empty() {
            bail!(
                "CLIP weight mapping was incomplete (missing: {:?}, errors: {:?})",
                result.missing,
                result.errors
            );
        }
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("failed to load CLIP tokenizer: {error}"))?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn image_embedding(&self, path: &Path) -> Result<Vec<f32>> {
        let image = image::open(path)
            .with_context(|| format!("failed to decode {} for CLIP", path.display()))?;
        let values = preprocess_image(&image);
        let tensor = Tensor::<Backend, 1>::from_floats(values.as_slice(), &self.device)
            .reshape([1, 3, IMAGE_SIZE, IMAGE_SIZE]);
        tensor_to_vector(self.model.encode_image(tensor))
    }

    fn text_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize semantic text: {error}"))?;
        let mut ids = encoding.get_ids().to_vec();
        if ids.len() > CONTEXT_LENGTH {
            ids.truncate(CONTEXT_LENGTH);
            ids[CONTEXT_LENGTH - 1] = EOT_TOKEN;
        }
        ids.resize(CONTEXT_LENGTH, EOT_TOKEN);
        let eot_index = ids
            .iter()
            .position(|token| *token == EOT_TOKEN)
            .unwrap_or(CONTEXT_LENGTH - 1);
        let ids: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
        let tokens = Tensor::<Backend, 1, Int>::from_ints(ids.as_slice(), &self.device)
            .reshape([1, CONTEXT_LENGTH]);
        tensor_to_vector(self.model.encode_text(tokens, eot_index))
    }
}

pub(super) fn analyze_missing(database: &Database, directory: &Path) -> Result<AiReport> {
    let key = model_key();
    let pending = database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT i.id, i.path FROM images i
             WHERE i.status='ready' AND NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.image_id=i.id AND e.model_id=?1
             ) ORDER BY i.path",
        )?;
        let rows = statement.query_map([&key], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                path_from_bytes(row.get_ref(1)?.as_blob()?),
            ))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })?;
    if pending.is_empty() {
        return Ok(AiReport::default());
    }
    let engine = ClipEngine::load(directory)?;
    let packs = list_label_packs(database)?;
    let prepared = prepare_packs(&engine, &packs)?;
    let mut report = AiReport::default();
    for (image_id, path) in pending {
        match engine.image_embedding(&path) {
            Ok(embedding) => {
                store_embedding(database, image_id, &key, &embedding)?;
                report.embedded += 1;
                score_image(database, image_id, &embedding, &prepared)?;
                report.scored += 1;
            }
            Err(error) => {
                report.failed += 1;
                report
                    .failures
                    .push(format!("{}: {error:#}", path.display()));
            }
        }
    }
    Ok(report)
}

pub(super) fn probe_rocm() -> Result<String> {
    let device = RocmDevice::default();
    let value = Tensor::<Backend, 1>::ones([1], &device)
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("ROCm tensor read failed: {error}"))?;
    if value != [1.0] {
        bail!("ROCm tensor verification returned unexpected data");
    }
    Ok(format!("Burn ROCm device {device:?}"))
}

pub(super) fn semantic_scores(
    database: &Database,
    directory: &Path,
    text: &str,
) -> Result<Vec<(i64, f32)>> {
    let text = text.trim();
    if text.is_empty() {
        bail!("semantic search text cannot be empty");
    }
    let engine = ClipEngine::load(directory)?;
    let query = engine.text_embedding(text)?;
    let key = model_key();
    let mut scored = database.with_connection(|connection| {
        let mut statement = connection
            .prepare("SELECT image_id, vector, dimension FROM embeddings WHERE model_id=?1")?;
        let rows = statement.query_map([key], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)? as usize,
            ))
        })?;
        let embeddings = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        let mut scores = Vec::with_capacity(embeddings.len());
        for (image_id, bytes, dimension) in embeddings {
            let embedding = decode_embedding(&bytes, dimension)?;
            scores.push((image_id, dot(&query, &embedding)));
        }
        Ok(scores)
    })?;
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(scored)
}

struct PreparedPack {
    id: i64,
    labels: Vec<(String, Vec<f32>)>,
}

fn prepare_packs(engine: &ClipEngine, packs: &[LabelPack]) -> Result<Vec<PreparedPack>> {
    packs
        .iter()
        .map(|pack| {
            let labels = pack
                .labels
                .iter()
                .map(|label| {
                    let embeddings = label
                        .prompts
                        .iter()
                        .map(|prompt| engine.text_embedding(prompt))
                        .collect::<Result<Vec<_>>>()?;
                    Ok((
                        label.name.clone(),
                        normalize_cpu(mean_vectors(&embeddings)?),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(PreparedPack {
                id: pack.id,
                labels,
            })
        })
        .collect()
}

fn score_image(
    database: &Database,
    image_id: i64,
    embedding: &[f32],
    packs: &[PreparedPack],
) -> Result<()> {
    database.with_transaction(|transaction| {
        for pack in packs {
            transaction.execute(
                "DELETE FROM label_scores WHERE image_id=?1 AND pack_id=?2",
                params![image_id, pack.id],
            )?;
            let logits: Vec<_> = pack
                .labels
                .iter()
                .map(|(_, text)| dot(embedding, text) * 100.0)
                .collect();
            let probabilities = softmax(&logits);
            for ((label, _), score) in pack.labels.iter().zip(probabilities) {
                transaction.execute(
                    "INSERT INTO label_scores(image_id, pack_id, label, score)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![image_id, pack.id, label, score],
                )?;
            }
        }
        Ok(())
    })
}

pub(super) fn rescore_label_packs(
    database: &Database,
    directory: &Path,
    wanted_pack: Option<&str>,
) -> Result<AiReport> {
    let mut packs = list_label_packs(database)?;
    if let Some(wanted) = wanted_pack {
        packs.retain(|pack| pack.name.eq_ignore_ascii_case(wanted));
        if packs.is_empty() {
            bail!("label pack not found: {wanted}");
        }
    }
    let engine = ClipEngine::load(directory)?;
    let prepared = prepare_packs(&engine, &packs)?;
    let key = model_key();
    let embeddings = database.with_connection(|connection| {
        let mut statement = connection.prepare(
            "SELECT image_id, vector, dimension FROM embeddings
             WHERE model_id=?1 AND normalized=1 ORDER BY image_id",
        )?;
        let rows = statement.query_map([key], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)? as usize,
            ))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })?;
    let mut report = AiReport::default();
    for (image_id, bytes, dimension) in embeddings {
        let result = decode_embedding(&bytes, dimension)
            .and_then(|embedding| score_image(database, image_id, &embedding, &prepared));
        match result {
            Ok(()) => report.scored += 1,
            Err(error) => {
                report.failed += 1;
                report.failures.push(format!("image {image_id}: {error:#}"));
            }
        }
    }
    Ok(report)
}

fn store_embedding(database: &Database, image_id: i64, key: &str, embedding: &[f32]) -> Result<()> {
    if embedding.len() != EMBEDDING_DIMENSION {
        bail!(
            "CLIP returned {} dimensions, expected {EMBEDDING_DIMENSION}",
            embedding.len()
        );
    }
    let bytes = encode_embedding(embedding);
    database.with_connection(|connection| {
        connection.execute(
            "INSERT INTO embeddings(image_id, model_id, dimension, vector, normalized, created_at)
             VALUES (?1, ?2, ?3, ?4, 1, ?5)
             ON CONFLICT(image_id, model_id) DO UPDATE SET
                dimension=excluded.dimension, vector=excluded.vector,
                normalized=1, created_at=excluded.created_at",
            params![
                image_id,
                key,
                EMBEDDING_DIMENSION as i64,
                bytes,
                chrono::Utc::now().timestamp_millis(),
            ],
        )?;
        Ok(())
    })
}

fn preprocess_image(image: &DynamicImage) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
    let resized =
        image.resize_to_fill(IMAGE_SIZE as u32, IMAGE_SIZE as u32, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let mut values = vec![0.0_f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let offset = y as usize * IMAGE_SIZE + x as usize;
        for channel in 0..3 {
            let value = f32::from(pixel.0[channel]) / 255.0;
            values[channel * IMAGE_SIZE * IMAGE_SIZE + offset] =
                (value - MEAN[channel]) / STD[channel];
        }
    }
    values
}

fn normalize<B: burn::tensor::backend::Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = (tensor.clone() * tensor.clone()).sum_dim(1).sqrt();
    tensor / norm
}

fn tensor_to_vector(tensor: Tensor<Backend, 2>) -> Result<Vec<f32>> {
    tensor
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("failed to read ROCm tensor: {error}"))
}

fn encode_embedding(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_embedding(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    if bytes.len() != dimension * 4 {
        bail!("embedding blob length does not match its dimension");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn mean_vectors(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    let first = vectors
        .first()
        .context("label must have at least one prompt")?;
    let mut mean = vec![0.0_f32; first.len()];
    for vector in vectors {
        if vector.len() != mean.len() {
            bail!("prompt embeddings have inconsistent dimensions");
        }
        for (target, value) in mean.iter_mut().zip(vector) {
            *target += value;
        }
    }
    for value in &mut mean {
        *value /= vectors.len() as f32;
    }
    Ok(mean)
}

fn normalize_cpu(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn embedding_blob_round_trips() {
        let input = vec![0.25, -0.5, 1.0];
        assert_eq!(
            decode_embedding(&encode_embedding(&input), 3).expect("decode"),
            input
        );
    }

    #[test]
    fn label_scoring_is_ranked_probability() {
        let scores = softmax(&[1.0, 3.0, 2.0]);
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires the pinned model and an AMD ROCm GPU"]
    fn rocm_clip_has_stable_embedding_dimensions_and_ranking() {
        let directory = PathBuf::from(
            std::env::var("BGM_CLIP_MODEL_DIR").expect("set BGM_CLIP_MODEL_DIR for gated test"),
        );
        let engine = ClipEngine::load(&directory).expect("load on ROCm");
        let image_directory = tempfile::tempdir().expect("tempdir");
        let image_path = image_directory.path().join("red.png");
        image::RgbImage::from_pixel(224, 224, image::Rgb([220, 30, 30]))
            .save(&image_path)
            .expect("test image");
        let image = engine
            .image_embedding(&image_path)
            .expect("image embedding");
        let cat = engine.text_embedding("a cat").expect("cat embedding");
        let mountain = engine
            .text_embedding("a mountain")
            .expect("mountain embedding");
        assert_eq!(image.len(), EMBEDDING_DIMENSION);
        assert!((dot(&image, &image) - 1.0).abs() < 1e-3);
        assert_eq!(cat.len(), EMBEDDING_DIMENSION);
        assert_eq!(mountain.len(), EMBEDDING_DIMENSION);
        assert!(dot(&cat, &cat) > dot(&cat, &mountain));
    }
}
