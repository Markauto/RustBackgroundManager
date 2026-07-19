use std::path::{Path, PathBuf};

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
use rusqlite::{Transaction, params, params_from_iter, types::Value};
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
// One model key plus 500 image IDs remains below SQLite's legacy 999-variable limit.
const CANDIDATE_QUERY_BATCH_SIZE: usize = 500;
const IMAGE_INFERENCE_BATCH_SIZE: usize = 8;
const EMBEDDING_WRITE_BATCH_SIZE: usize = 32;
const SCORE_WRITE_BATCH_SIZE: usize = 64;

type PendingImage = (i64, PathBuf);
type StoredEmbedding = (i64, Vec<u8>, usize);

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
        self.image_embeddings(std::slice::from_ref(&path))?
            .pop()
            .context("CLIP returned no embedding for one image")
    }

    fn image_embeddings(&self, paths: &[&Path]) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(!paths.is_empty(), "CLIP image batch cannot be empty");
        let values_per_image = 3 * IMAGE_SIZE * IMAGE_SIZE;
        let mut values = Vec::with_capacity(paths.len() * values_per_image);
        for path in paths {
            let image = image::open(path)
                .with_context(|| format!("failed to decode {} for CLIP", path.display()))?;
            values.extend(preprocess_image(&image));
        }
        let tensor = Tensor::<Backend, 1>::from_floats(values.as_slice(), &self.device).reshape([
            paths.len(),
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ]);
        split_embedding_batch(
            tensor_to_vector(self.model.encode_image(tensor))?,
            paths.len(),
        )
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
    let pending = load_pending_images(database, &key, None)?;
    if pending.is_empty() {
        return Ok(AiReport::default());
    }
    let engine = ClipEngine::load(directory)?;
    analyze_pending_with_engine(database, &engine, &key, pending)
}

fn analyze_pending_with_engine(
    database: &Database,
    engine: &ClipEngine,
    key: &str,
    pending: Vec<PendingImage>,
) -> Result<AiReport> {
    let packs = list_label_packs(database)?;
    let prepared = prepare_packs(engine, &packs)?;
    let mut report = AiReport::default();
    let mut completed = Vec::with_capacity(EMBEDDING_WRITE_BATCH_SIZE);
    for batch in pending.chunks(IMAGE_INFERENCE_BATCH_SIZE) {
        let paths = batch
            .iter()
            .map(|(_, path)| path.as_path())
            .collect::<Vec<_>>();
        match engine.image_embeddings(&paths) {
            Ok(embeddings) => {
                for ((image_id, path), embedding) in batch.iter().zip(embeddings) {
                    completed.push((*image_id, path.clone(), embedding));
                }
            }
            Err(_) => {
                // A corrupt image or a batch-size-specific GPU failure should
                // not prevent healthy images in the same batch from completing.
                for (image_id, path) in batch {
                    match engine.image_embedding(path) {
                        Ok(embedding) => completed.push((*image_id, path.clone(), embedding)),
                        Err(error) => {
                            report.failed += 1;
                            report
                                .failures
                                .push(format!("{}: {error:#}", path.display()));
                        }
                    }
                }
            }
        }
        if completed.len() >= EMBEDDING_WRITE_BATCH_SIZE {
            persist_embedding_batch(database, key, &completed, &prepared, &mut report);
            completed.clear();
        }
    }
    persist_embedding_batch(database, key, &completed, &prepared, &mut report);
    Ok(report)
}

fn persist_embedding_batch(
    database: &Database,
    key: &str,
    embeddings: &[(i64, PathBuf, Vec<f32>)],
    packs: &[PreparedPack],
    report: &mut AiReport,
) {
    if embeddings.is_empty() {
        return;
    }
    let batched = database.with_transaction(|transaction| {
        for (image_id, _, embedding) in embeddings {
            replace_embedding_and_scores(transaction, *image_id, key, embedding, packs)?;
        }
        Ok(())
    });
    if batched.is_ok() {
        report.embedded += embeddings.len();
        report.scored += embeddings.len();
        return;
    }

    // Preserve per-image failure isolation if one record makes the batch roll back.
    for (image_id, path, embedding) in embeddings {
        match store_embedding_and_scores(database, *image_id, key, embedding, packs) {
            Ok(()) => {
                report.embedded += 1;
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
}

fn load_pending_images(
    database: &Database,
    key: &str,
    image_ids: Option<&[i64]>,
) -> Result<Vec<PendingImage>> {
    let Some(image_ids) = image_ids else {
        return database.with_connection(|connection| {
            let sql = format!(
                "SELECT i.id, i.path FROM images i
                 WHERE i.status='ready' AND NOT EXISTS (
                    SELECT 1 FROM embeddings e
                    WHERE e.image_id=i.id AND e.model_id=?1 AND e.normalized=1
                      AND e.dimension={EMBEDDING_DIMENSION}
                      AND length(e.vector)={}
                 ) ORDER BY i.path",
                EMBEDDING_DIMENSION * 4
            );
            let mut statement = connection.prepare(&sql)?;
            let rows = statement.query_map([key], pending_image_from_row)?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        });
    };
    if image_ids.is_empty() {
        return Ok(Vec::new());
    }

    database.with_connection(|connection| {
        let mut pending = Vec::new();
        for batch in image_ids.chunks(CANDIDATE_QUERY_BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!(
                "SELECT i.id, i.path FROM images i
                 WHERE i.status='ready' AND NOT EXISTS (
                    SELECT 1 FROM embeddings e
                    WHERE e.image_id=i.id AND e.model_id=? AND e.normalized=1
                      AND e.dimension={EMBEDDING_DIMENSION}
                      AND length(e.vector)={}
                 ) AND i.id IN ({placeholders}) ORDER BY i.path",
                EMBEDDING_DIMENSION * 4
            );
            let parameters = std::iter::once(Value::Text(key.to_owned()))
                .chain(batch.iter().copied().map(Value::Integer));
            let mut statement = connection.prepare(&sql)?;
            let rows = statement.query_map(params_from_iter(parameters), pending_image_from_row)?;
            pending.extend(rows.collect::<rusqlite::Result<Vec<_>>>()?);
        }
        pending.sort_by(|left, right| left.1.cmp(&right.1));
        Ok(pending)
    })
}

fn pending_image_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PendingImage> {
    Ok((row.get(0)?, path_from_bytes(row.get_ref(1)?.as_blob()?)))
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
    semantic_scores_inner(database, directory, text, None)
}

pub(super) fn semantic_scores_for_images(
    database: &Database,
    directory: &Path,
    text: &str,
    image_ids: &[i64],
) -> Result<Vec<(i64, f32)>> {
    let text = validated_semantic_text(text)?;
    let key = model_key();
    let pending = load_pending_images(database, &key, Some(image_ids))?;
    if pending.is_empty() {
        return semantic_scores_inner(database, directory, text, Some(image_ids));
    }

    let engine = ClipEngine::load(directory)?;
    let _ = analyze_pending_with_engine(database, &engine, &key, pending)?;
    let embeddings = load_semantic_embeddings(database, &key, Some(image_ids))?;
    score_semantic_embeddings(&engine, text, embeddings)
}

fn semantic_scores_inner(
    database: &Database,
    directory: &Path,
    text: &str,
    image_ids: Option<&[i64]>,
) -> Result<Vec<(i64, f32)>> {
    let text = validated_semantic_text(text)?;
    let key = model_key();
    let embeddings = load_semantic_embeddings(database, &key, image_ids)?;
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    let engine = ClipEngine::load(directory)?;
    score_semantic_embeddings(&engine, text, embeddings)
}

fn validated_semantic_text(text: &str) -> Result<&str> {
    let text = text.trim();
    if text.is_empty() {
        bail!("semantic search text cannot be empty");
    }
    Ok(text)
}

fn score_semantic_embeddings(
    engine: &ClipEngine,
    text: &str,
    embeddings: Vec<StoredEmbedding>,
) -> Result<Vec<(i64, f32)>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    let query = engine.text_embedding(text)?;
    let mut scored = Vec::with_capacity(embeddings.len());
    for (image_id, bytes, dimension) in embeddings {
        let embedding = decode_model_embedding(&bytes, dimension)?;
        scored.push((image_id, dot(&query, &embedding)));
    }
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(scored)
}

fn load_semantic_embeddings(
    database: &Database,
    key: &str,
    image_ids: Option<&[i64]>,
) -> Result<Vec<StoredEmbedding>> {
    let Some(image_ids) = image_ids else {
        return database.with_connection(|connection| {
            let sql = format!(
                "SELECT image_id, vector, dimension FROM embeddings
                 WHERE model_id=?1 AND normalized=1
                   AND dimension={EMBEDDING_DIMENSION} AND length(vector)={}
                 ORDER BY image_id",
                EMBEDDING_DIMENSION * 4
            );
            let mut statement = connection.prepare(&sql)?;
            let rows = statement.query_map([key], embedding_from_row)?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        });
    };
    if image_ids.is_empty() {
        return Ok(Vec::new());
    }

    database.with_connection(|connection| {
        let mut embeddings = Vec::new();
        for batch in image_ids.chunks(CANDIDATE_QUERY_BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!(
                "SELECT image_id, vector, dimension FROM embeddings
                 WHERE model_id=? AND normalized=1
                   AND dimension={EMBEDDING_DIMENSION} AND length(vector)={}
                   AND image_id IN ({placeholders}) ORDER BY image_id",
                EMBEDDING_DIMENSION * 4
            );
            let parameters = std::iter::once(Value::Text(key.to_owned()))
                .chain(batch.iter().copied().map(Value::Integer));
            let mut statement = connection.prepare(&sql)?;
            let rows = statement.query_map(params_from_iter(parameters), embedding_from_row)?;
            embeddings.extend(rows.collect::<rusqlite::Result<Vec<_>>>()?);
        }
        Ok(embeddings)
    })
}

fn embedding_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredEmbedding> {
    Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? as usize))
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
    database.with_transaction(|transaction| replace_scores(transaction, image_id, embedding, packs))
}

fn replace_scores(
    transaction: &Transaction<'_>,
    image_id: i64,
    embedding: &[f32],
    packs: &[PreparedPack],
) -> Result<()> {
    let mut delete_scores =
        transaction.prepare_cached("DELETE FROM label_scores WHERE image_id=?1 AND pack_id=?2")?;
    let mut insert_score = transaction.prepare_cached(
        "INSERT INTO label_scores(image_id, pack_id, label, score)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for pack in packs {
        delete_scores.execute(params![image_id, pack.id])?;
        let logits: Vec<_> = pack
            .labels
            .iter()
            .map(|(_, text)| dot(embedding, text) * 100.0)
            .collect();
        let probabilities = softmax(&logits);
        for ((label, _), score) in pack.labels.iter().zip(probabilities) {
            insert_score.execute(params![image_id, pack.id, label, score])?;
        }
    }
    Ok(())
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
    let mut report = AiReport::default();
    let mut after_image_id = None;
    loop {
        let embeddings = load_rescore_embedding_page(database, &key, after_image_id)?;
        let Some((last_image_id, _, _)) = embeddings.last() else {
            break;
        };
        after_image_id = Some(*last_image_id);
        let mut pending = Vec::with_capacity(embeddings.len());
        for (image_id, bytes, dimension) in embeddings {
            match decode_model_embedding(&bytes, dimension) {
                Ok(embedding) => pending.push((image_id, embedding)),
                Err(error) => {
                    report.failed += 1;
                    report.failures.push(format!("image {image_id}: {error:#}"));
                }
            }
        }
        persist_score_batch(database, &pending, &prepared, &mut report);
    }
    Ok(report)
}

fn load_rescore_embedding_page(
    database: &Database,
    key: &str,
    after_image_id: Option<i64>,
) -> Result<Vec<StoredEmbedding>> {
    database.with_connection(|connection| {
        let (sql, parameters): (&str, Vec<Value>) = if let Some(image_id) = after_image_id {
            (
                "SELECT image_id, vector, dimension FROM embeddings
                 WHERE model_id=?1 AND normalized=1 AND image_id>?2
                 ORDER BY image_id LIMIT ?3",
                vec![
                    Value::Text(key.to_owned()),
                    Value::Integer(image_id),
                    Value::Integer(SCORE_WRITE_BATCH_SIZE as i64),
                ],
            )
        } else {
            (
                "SELECT image_id, vector, dimension FROM embeddings
                 WHERE model_id=?1 AND normalized=1
                 ORDER BY image_id LIMIT ?2",
                vec![
                    Value::Text(key.to_owned()),
                    Value::Integer(SCORE_WRITE_BATCH_SIZE as i64),
                ],
            )
        };
        let mut statement = connection.prepare(sql)?;
        let rows = statement.query_map(params_from_iter(parameters), embedding_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

fn persist_score_batch(
    database: &Database,
    embeddings: &[(i64, Vec<f32>)],
    packs: &[PreparedPack],
    report: &mut AiReport,
) {
    if embeddings.is_empty() {
        return;
    }
    let batched = database.with_transaction(|transaction| {
        for (image_id, embedding) in embeddings {
            replace_scores(transaction, *image_id, embedding, packs)?;
        }
        Ok(())
    });
    if batched.is_ok() {
        report.scored += embeddings.len();
        return;
    }

    // Preserve per-image failure isolation if one record makes the batch roll back.
    for (image_id, embedding) in embeddings {
        match score_image(database, *image_id, embedding, packs) {
            Ok(()) => report.scored += 1,
            Err(error) => {
                report.failed += 1;
                report.failures.push(format!("image {image_id}: {error:#}"));
            }
        }
    }
}

fn store_embedding_and_scores(
    database: &Database,
    image_id: i64,
    key: &str,
    embedding: &[f32],
    packs: &[PreparedPack],
) -> Result<()> {
    database.with_transaction(|transaction| {
        replace_embedding_and_scores(transaction, image_id, key, embedding, packs)
    })
}

fn replace_embedding_and_scores(
    transaction: &Transaction<'_>,
    image_id: i64,
    key: &str,
    embedding: &[f32],
    packs: &[PreparedPack],
) -> Result<()> {
    validate_model_embedding(embedding)?;
    let bytes = encode_embedding(embedding);
    transaction.execute(
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
    replace_scores(transaction, image_id, embedding, packs)
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

fn split_embedding_batch(values: Vec<f32>, batch_size: usize) -> Result<Vec<Vec<f32>>> {
    anyhow::ensure!(batch_size != 0, "CLIP embedding batch cannot be empty");
    anyhow::ensure!(
        values.len() == batch_size * EMBEDDING_DIMENSION,
        "CLIP returned {} values for a batch of {batch_size}; expected {}",
        values.len(),
        batch_size * EMBEDDING_DIMENSION
    );
    Ok(values
        .chunks_exact(EMBEDDING_DIMENSION)
        .map(<[f32]>::to_vec)
        .collect())
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

fn decode_model_embedding(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    if dimension != EMBEDDING_DIMENSION {
        bail!("stored CLIP embedding has {dimension} dimensions, expected {EMBEDDING_DIMENSION}");
    }
    let embedding = decode_embedding(bytes, dimension)?;
    validate_model_embedding(&embedding)?;
    Ok(embedding)
}

fn validate_model_embedding(embedding: &[f32]) -> Result<()> {
    if embedding.len() != EMBEDDING_DIMENSION {
        bail!(
            "CLIP embedding has {} dimensions, expected {EMBEDDING_DIMENSION}",
            embedding.len()
        );
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        bail!("CLIP embedding contains a non-finite value");
    }
    let norm_squared = embedding.iter().map(|value| value * value).sum::<f32>();
    if (norm_squared - 1.0).abs() > 0.01 {
        bail!("CLIP embedding is not normalized (squared norm {norm_squared})");
    }
    Ok(())
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

    fn unit_embedding() -> Vec<f32> {
        let value = 1.0 / (EMBEDDING_DIMENSION as f32).sqrt();
        vec![value; EMBEDDING_DIMENSION]
    }

    #[test]
    fn embedding_blob_round_trips() {
        let input = vec![0.25, -0.5, 1.0];
        assert_eq!(
            decode_embedding(&encode_embedding(&input), 3).expect("decode"),
            input
        );
        assert!(decode_model_embedding(&encode_embedding(&input), 3).is_err());
        assert!(
            decode_model_embedding(
                &encode_embedding(&vec![0.25; EMBEDDING_DIMENSION]),
                EMBEDDING_DIMENSION,
            )
            .is_err()
        );
        let mut non_finite = unit_embedding();
        non_finite[0] = f32::NAN;
        assert!(
            decode_model_embedding(&encode_embedding(&non_finite), EMBEDDING_DIMENSION).is_err()
        );
    }

    #[test]
    fn batched_embedding_output_is_split_by_image() {
        let mut values = vec![0.0; EMBEDDING_DIMENSION * 2];
        values[0] = 1.0;
        values[EMBEDDING_DIMENSION] = 2.0;

        let embeddings = split_embedding_batch(values, 2).expect("split batch");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0][0], 1.0);
        assert_eq!(embeddings[1][0], 2.0);
        assert!(split_embedding_batch(vec![0.0; EMBEDDING_DIMENSION], 2).is_err());
    }

    #[test]
    fn label_scoring_is_ranked_probability() {
        let scores = softmax(&[1.0, 3.0, 2.0]);
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn candidate_embedding_queries_are_scoped_and_batched() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let key = model_key();
        let vector = encode_embedding(&unit_embedding());
        let image_ids = database
            .with_transaction(|transaction| {
                let mut image_ids = Vec::new();
                for index in 0..(CANDIDATE_QUERY_BATCH_SIZE + 2) {
                    let image_path = source_path.join(format!("wallpaper-{index}.png"));
                    transaction.execute(
                        "INSERT INTO images(
                            source_id, path, size, modified_ns, status, discovered_at, updated_at
                         ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                        params![source.id, crate::db::path_bytes(&image_path)],
                    )?;
                    let image_id = transaction.last_insert_rowid();
                    image_ids.push(image_id);
                    transaction.execute(
                        "INSERT INTO embeddings(
                            image_id, model_id, dimension, vector, normalized, created_at
                         ) VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                        params![
                            image_id,
                            key,
                            EMBEDDING_DIMENSION as i64,
                            vector,
                            i64::from(index < CANDIDATE_QUERY_BATCH_SIZE + 1),
                        ],
                    )?;
                }
                Ok(image_ids)
            })
            .expect("fixture data");

        let requested = &image_ids[..CANDIDATE_QUERY_BATCH_SIZE + 1];
        let embeddings = load_semantic_embeddings(&database, &key, Some(requested))
            .expect("candidate embeddings");
        assert_eq!(
            embeddings
                .iter()
                .map(|(image_id, _, _)| *image_id)
                .collect::<Vec<_>>(),
            requested
        );
        let mut after_image_id = None;
        let mut paged_count = 0;
        loop {
            let page = load_rescore_embedding_page(&database, &key, after_image_id)
                .expect("rescore embedding page");
            let Some((last_image_id, _, _)) = page.last() else {
                break;
            };
            assert!(page.len() <= SCORE_WRITE_BATCH_SIZE);
            after_image_id = Some(*last_image_id);
            paged_count += page.len();
        }
        assert_eq!(paged_count, requested.len());
        assert!(
            load_pending_images(&database, &key, Some(requested))
                .expect("embedded candidates")
                .is_empty()
        );

        let unnormalized_id = *image_ids.last().expect("unnormalized image");
        assert!(
            load_semantic_embeddings(&database, &key, Some(&[unnormalized_id]))
                .expect("unnormalized candidate")
                .is_empty()
        );
        assert_eq!(
            load_pending_images(&database, &key, Some(&[unnormalized_id]))
                .expect("pending candidate"),
            vec![(
                unnormalized_id,
                source_path.join(format!("wallpaper-{}.png", CANDIDATE_QUERY_BATCH_SIZE + 1)),
            )]
        );
        database
            .with_connection(|connection| {
                connection.execute(
                    "UPDATE embeddings SET normalized=1, dimension=1 WHERE image_id=?1",
                    [unnormalized_id],
                )?;
                Ok(())
            })
            .expect("malformed embedding");
        assert_eq!(
            load_pending_images(&database, &key, Some(&[unnormalized_id]))
                .expect("malformed candidate"),
            vec![(
                unnormalized_id,
                source_path.join(format!("wallpaper-{}.png", CANDIDATE_QUERY_BATCH_SIZE + 1)),
            )]
        );
        assert!(
            load_semantic_embeddings(&database, &key, Some(&[unnormalized_id]))
                .expect("malformed semantic embedding")
                .is_empty()
        );
    }

    #[test]
    fn embedding_analysis_persists_more_than_one_write_batch() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let (image_ids, pack_id) = database
            .with_transaction(|transaction| {
                let mut image_ids = Vec::new();
                for index in 0..(EMBEDDING_WRITE_BATCH_SIZE + 1) {
                    let image_path = source_path.join(format!("wallpaper-{index}.png"));
                    transaction.execute(
                        "INSERT INTO images(
                            source_id, path, size, modified_ns, status, discovered_at, updated_at
                         ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                        params![source.id, crate::db::path_bytes(&image_path)],
                    )?;
                    image_ids.push(transaction.last_insert_rowid());
                }
                let pack_id = transaction.query_row(
                    "SELECT id FROM label_packs WHERE name='mood'",
                    [],
                    |row| row.get(0),
                )?;
                Ok((image_ids, pack_id))
            })
            .expect("fixture data");
        let pack = PreparedPack {
            id: pack_id,
            labels: vec![("valid".into(), vec![0.5; EMBEDDING_DIMENSION])],
        };
        let embeddings = image_ids
            .iter()
            .map(|image_id| {
                (
                    *image_id,
                    source_path.join(format!("wallpaper-{image_id}.png")),
                    unit_embedding(),
                )
            })
            .collect::<Vec<_>>();
        let mut report = AiReport::default();
        for batch in embeddings.chunks(EMBEDDING_WRITE_BATCH_SIZE) {
            persist_embedding_batch(
                &database,
                &model_key(),
                batch,
                std::slice::from_ref(&pack),
                &mut report,
            );
        }

        assert_eq!(report.embedded, EMBEDDING_WRITE_BATCH_SIZE + 1);
        assert_eq!(report.scored, EMBEDDING_WRITE_BATCH_SIZE + 1);
        assert_eq!(report.failed, 0);
        let counts = database
            .with_connection(|connection| {
                connection
                    .query_row(
                        "SELECT
                            (SELECT COUNT(*) FROM embeddings),
                            (SELECT COUNT(*) FROM label_scores)",
                        [],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .map_err(Into::into)
            })
            .expect("stored counts");
        assert_eq!(
            counts,
            (
                (EMBEDDING_WRITE_BATCH_SIZE + 1) as i64,
                (EMBEDDING_WRITE_BATCH_SIZE + 1) as i64,
            )
        );
    }

    #[test]
    fn embedding_batch_falls_back_to_isolate_a_bad_image() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let (image_ids, pack_id) = database
            .with_transaction(|transaction| {
                let mut image_ids = Vec::new();
                for index in 0..2 {
                    let image_path = source_path.join(format!("wallpaper-{index}.png"));
                    transaction.execute(
                        "INSERT INTO images(
                            source_id, path, size, modified_ns, status, discovered_at, updated_at
                         ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                        params![source.id, crate::db::path_bytes(&image_path)],
                    )?;
                    image_ids.push(transaction.last_insert_rowid());
                }
                let pack_id = transaction.query_row(
                    "SELECT id FROM label_packs WHERE name='mood'",
                    [],
                    |row| row.get(0),
                )?;
                Ok((image_ids, pack_id))
            })
            .expect("fixture data");
        let embedding = unit_embedding();
        let embeddings = vec![
            (image_ids[0], PathBuf::from("first.png"), embedding.clone()),
            (i64::MAX, PathBuf::from("invalid.png"), embedding.clone()),
            (image_ids[1], PathBuf::from("second.png"), embedding),
        ];
        let pack = PreparedPack {
            id: pack_id,
            labels: vec![("valid".into(), vec![0.5; EMBEDDING_DIMENSION])],
        };
        let mut report = AiReport::default();

        persist_embedding_batch(
            &database,
            &model_key(),
            &embeddings,
            std::slice::from_ref(&pack),
            &mut report,
        );

        assert_eq!(report.embedded, 2);
        assert_eq!(report.scored, 2);
        assert_eq!(report.failed, 1);
        assert!(report.failures[0].starts_with("invalid.png:"));
        let stored = database
            .with_connection(|connection| {
                let mut statement =
                    connection.prepare("SELECT image_id FROM embeddings ORDER BY image_id")?;
                let rows = statement.query_map([], |row| row.get::<_, i64>(0))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(Into::into)
            })
            .expect("stored embeddings");
        assert_eq!(stored, image_ids);
    }

    #[test]
    fn label_rescoring_persists_more_than_one_write_batch() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let (image_ids, pack_id) = database
            .with_transaction(|transaction| {
                let mut image_ids = Vec::new();
                for index in 0..(SCORE_WRITE_BATCH_SIZE + 1) {
                    let image_path = source_path.join(format!("wallpaper-{index}.png"));
                    transaction.execute(
                        "INSERT INTO images(
                            source_id, path, size, modified_ns, status, discovered_at, updated_at
                         ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                        params![source.id, crate::db::path_bytes(&image_path)],
                    )?;
                    image_ids.push(transaction.last_insert_rowid());
                }
                let pack_id = transaction.query_row(
                    "SELECT id FROM label_packs WHERE name='mood'",
                    [],
                    |row| row.get(0),
                )?;
                Ok((image_ids, pack_id))
            })
            .expect("fixture data");
        let embedding = unit_embedding();
        let pack = PreparedPack {
            id: pack_id,
            labels: vec![
                ("first".into(), vec![0.5; EMBEDDING_DIMENSION]),
                ("second".into(), vec![-0.5; EMBEDDING_DIMENSION]),
            ],
        };
        let mut report = AiReport::default();
        let embeddings = image_ids
            .iter()
            .map(|image_id| (*image_id, embedding.clone()))
            .collect::<Vec<_>>();
        for batch in embeddings.chunks(SCORE_WRITE_BATCH_SIZE) {
            persist_score_batch(&database, batch, std::slice::from_ref(&pack), &mut report);
        }

        assert_eq!(report.scored, SCORE_WRITE_BATCH_SIZE + 1);
        assert_eq!(report.failed, 0);
        let score_count = database
            .with_connection(|connection| {
                connection
                    .query_row("SELECT COUNT(*) FROM label_scores", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .map_err(Into::into)
            })
            .expect("score count");
        assert_eq!(score_count, ((SCORE_WRITE_BATCH_SIZE + 1) * 2) as i64);
    }

    #[test]
    fn label_rescoring_falls_back_to_isolate_a_bad_image() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let (image_ids, pack_id) = database
            .with_transaction(|transaction| {
                let mut image_ids = Vec::new();
                for index in 0..2 {
                    let image_path = source_path.join(format!("wallpaper-{index}.png"));
                    transaction.execute(
                        "INSERT INTO images(
                            source_id, path, size, modified_ns, status, discovered_at, updated_at
                         ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                        params![source.id, crate::db::path_bytes(&image_path)],
                    )?;
                    image_ids.push(transaction.last_insert_rowid());
                }
                let pack_id = transaction.query_row(
                    "SELECT id FROM label_packs WHERE name='mood'",
                    [],
                    |row| row.get(0),
                )?;
                Ok((image_ids, pack_id))
            })
            .expect("fixture data");
        let embedding = unit_embedding();
        let embeddings = vec![
            (image_ids[0], embedding.clone()),
            (i64::MAX, embedding.clone()),
            (image_ids[1], embedding),
        ];
        let pack = PreparedPack {
            id: pack_id,
            labels: vec![("valid".into(), vec![0.5; EMBEDDING_DIMENSION])],
        };
        let mut report = AiReport::default();

        persist_score_batch(
            &database,
            &embeddings,
            std::slice::from_ref(&pack),
            &mut report,
        );

        assert_eq!(report.scored, 2);
        assert_eq!(report.failed, 1);
        assert!(report.failures[0].starts_with(&format!("image {}:", i64::MAX)));
        let scored_images = database
            .with_connection(|connection| {
                let mut statement =
                    connection.prepare("SELECT image_id FROM label_scores ORDER BY image_id")?;
                let rows = statement.query_map([], |row| row.get::<_, i64>(0))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(Into::into)
            })
            .expect("scored images");
        assert_eq!(scored_images, image_ids);
    }

    #[test]
    fn embedding_and_scores_commit_atomically() {
        let directory = tempfile::tempdir().expect("tempdir");
        let database = Database::open(directory.path().join("catalog.sqlite3")).expect("database");
        let source_path = directory.path().join("images");
        std::fs::create_dir(&source_path).expect("source directory");
        let source = database.add_source(&source_path).expect("source");
        let image_path = source_path.join("wallpaper.png");
        let (image_id, pack_id) = database
            .with_connection(|connection| {
                connection.execute(
                    "INSERT INTO images(
                        source_id, path, size, modified_ns, status, discovered_at, updated_at
                     ) VALUES (?1, ?2, 0, 0, 'ready', 0, 0)",
                    params![source.id, crate::db::path_bytes(&image_path)],
                )?;
                let image_id = connection.last_insert_rowid();
                let pack_id = connection.query_row(
                    "SELECT id FROM label_packs WHERE name='mood'",
                    [],
                    |row| row.get(0),
                )?;
                Ok((image_id, pack_id))
            })
            .expect("fixture data");
        let embedding = unit_embedding();
        let duplicate_labels = PreparedPack {
            id: pack_id,
            labels: vec![
                ("duplicate".into(), vec![0.5; EMBEDDING_DIMENSION]),
                ("duplicate".into(), vec![-0.5; EMBEDDING_DIMENSION]),
            ],
        };

        assert!(
            store_embedding_and_scores(
                &database,
                image_id,
                &model_key(),
                &embedding,
                &[duplicate_labels],
            )
            .is_err()
        );
        let counts = database
            .with_connection(|connection| {
                connection
                    .query_row(
                        "SELECT
                            (SELECT COUNT(*) FROM embeddings WHERE image_id=?1),
                            (SELECT COUNT(*) FROM label_scores WHERE image_id=?1)",
                        [image_id],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .map_err(Into::into)
            })
            .expect("rollback counts");
        assert_eq!(counts, (0, 0));

        let valid_pack = PreparedPack {
            id: pack_id,
            labels: vec![
                ("first".into(), vec![0.5; EMBEDDING_DIMENSION]),
                ("second".into(), vec![-0.5; EMBEDDING_DIMENSION]),
            ],
        };
        store_embedding_and_scores(&database, image_id, &model_key(), &embedding, &[valid_pack])
            .expect("atomic store");
        let counts = database
            .with_connection(|connection| {
                connection
                    .query_row(
                        "SELECT
                            (SELECT COUNT(*) FROM embeddings WHERE image_id=?1),
                            (SELECT COUNT(*) FROM label_scores WHERE image_id=?1)",
                        [image_id],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .map_err(Into::into)
            })
            .expect("stored counts");
        assert_eq!(counts, (1, 2));
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
