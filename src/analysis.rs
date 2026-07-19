use std::{cmp::Ordering, path::Path};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView as _, ImageReader, imageops::FilterType};
use serde::{Deserialize, Serialize};

use crate::config::{AnalysisConfig, ImportConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    Landscape,
    Portrait,
    Square,
}

impl Orientation {
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        match width.cmp(&height) {
            Ordering::Greater => Self::Landscape,
            Ordering::Less => Self::Portrait,
            Ordering::Equal => Self::Square,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Landscape => "landscape",
            Self::Portrait => "portrait",
            Self::Square => "square",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaletteColor {
    pub oklab: Oklab,
    pub proportion: f32,
    pub hex: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageAnalysis {
    pub width: u32,
    pub height: u32,
    pub ratio: f64,
    pub orientation: Orientation,
    pub common_ratio: Option<String>,
    pub palette: Vec<PaletteColor>,
    pub dominant_hex: String,
    pub dominant_name: String,
    pub luminance: f32,
    pub saturation: f32,
    pub contrast: f32,
    pub light_dark: LightDark,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightDark {
    Light,
    Dark,
}

impl LightDark {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }
}

pub fn probe_dimensions(path: &Path) -> Result<(u32, u32)> {
    let reader = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("could not determine image format for {}", path.display()))?;
    reader
        .into_dimensions()
        .with_context(|| format!("could not read dimensions for {}", path.display()))
}

pub const fn within_import_bounds(width: u32, height: u32, bounds: &ImportConfig) -> bool {
    if let Some(minimum) = bounds.min_width
        && width < minimum
    {
        return false;
    }
    if let Some(maximum) = bounds.max_width
        && width > maximum
    {
        return false;
    }
    if let Some(minimum) = bounds.min_height
        && height < minimum
    {
        return false;
    }
    if let Some(maximum) = bounds.max_height
        && height > maximum
    {
        return false;
    }
    true
}

pub fn analyze_image(
    path: &Path,
    config: &AnalysisConfig,
) -> Result<(ImageAnalysis, DynamicImage)> {
    let image = decode_image(path)?;
    let analysis = analyze_pixels(&image, config)?;
    Ok((analysis, image))
}

pub fn decode_image(path: &Path) -> Result<DynamicImage> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?;
    Ok(image)
}

pub fn analyze_pixels(image: &DynamicImage, config: &AnalysisConfig) -> Result<ImageAnalysis> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        bail!("image has zero width or height");
    }
    // Palette extraction does not benefit from evaluating every source pixel.
    // A fixed 64px sample keeps the result deterministic while making large
    // wallpaper libraries practical to scan.
    let sample = image.resize(64, 64, FilterType::Triangle).to_rgba8();
    let mut points = Vec::with_capacity(sample.width() as usize * sample.height() as usize);
    let mut luminances = Vec::with_capacity(points.capacity());
    let mut saturations = Vec::with_capacity(points.capacity());
    for pixel in sample.pixels() {
        if pixel.0[3] < 16 {
            continue;
        }
        let rgb = [pixel.0[0], pixel.0[1], pixel.0[2]];
        points.push(srgb8_to_oklab(rgb));
        let linear = rgb.map(|channel| srgb_to_linear(f32::from(channel) / 255.0));
        luminances.push(relative_luminance(linear));
        saturations.push(hsv_saturation(rgb));
    }
    if points.is_empty() {
        bail!("image contains no visible pixels");
    }

    let palette = kmeans_palette(&points, usize::from(config.palette_colors));
    let dominant = palette
        .first()
        .context("palette extraction returned no colours")?;
    let dominant_hex = dominant.hex.clone();
    let dominant_name = dominant.name.clone();
    let luminance = mean(&luminances);
    let saturation = mean(&saturations);
    let contrast = standard_deviation(&luminances, luminance);
    Ok(ImageAnalysis {
        width,
        height,
        ratio: f64::from(width) / f64::from(height),
        orientation: Orientation::from_dimensions(width, height),
        common_ratio: nearest_common_ratio(width, height, config.common_ratio_tolerance),
        palette,
        dominant_hex,
        dominant_name,
        luminance,
        saturation,
        contrast,
        light_dark: classify_light_dark(luminance, config.dark_threshold),
    })
}

pub fn write_thumbnail(image: &DynamicImage, destination: &Path, long_edge: u32) -> Result<()> {
    let parent = destination
        .parent()
        .context("thumbnail destination has no parent")?;
    std::fs::create_dir_all(parent)?;
    let thumbnail = image.thumbnail(long_edge, long_edge);
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut temporary, 88);
        encoder.encode_image(&thumbnail.to_rgb8())?;
    }
    use std::io::Write as _;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(destination)
        .map_err(|error| error.error)?;
    Ok(())
}

pub fn nearest_common_ratio(width: u32, height: u32, tolerance: f32) -> Option<String> {
    if width == 0 || height == 0 {
        return None;
    }
    const RATIOS: [(&str, f64); 15] = [
        ("9:32", 9.0 / 32.0),
        ("9:21", 9.0 / 21.0),
        ("9:16", 9.0 / 16.0),
        ("10:16", 10.0 / 16.0),
        ("2:3", 2.0 / 3.0),
        ("3:4", 3.0 / 4.0),
        ("4:5", 4.0 / 5.0),
        ("1:1", 1.0),
        ("5:4", 5.0 / 4.0),
        ("4:3", 4.0 / 3.0),
        ("3:2", 3.0 / 2.0),
        ("16:10", 16.0 / 10.0),
        ("16:9", 16.0 / 9.0),
        ("21:9", 21.0 / 9.0),
        ("32:9", 32.0 / 9.0),
    ];
    let actual = f64::from(width) / f64::from(height);
    RATIOS
        .into_iter()
        .map(|(name, ratio)| (name, ratio, (actual - ratio).abs() / ratio))
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .filter(|(_, _, error)| *error <= f64::from(tolerance))
        .map(|(name, _, _)| name.to_owned())
}

pub const fn classify_light_dark(luminance: f32, threshold: f32) -> LightDark {
    if luminance >= threshold {
        LightDark::Light
    } else {
        LightDark::Dark
    }
}

pub fn colour_distance(left: Oklab, right: Oklab) -> f32 {
    ((left.l - right.l).powi(2) + (left.a - right.a).powi(2) + (left.b - right.b).powi(2)).sqrt()
}

pub fn parse_hex_colour(value: &str) -> Result<Oklab> {
    let value = value.strip_prefix('#').unwrap_or(value);
    if value.len() != 6 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("colour must be a six-digit hex value, such as #336699");
    }
    let bytes = hex::decode(value)?;
    Ok(srgb8_to_oklab([bytes[0], bytes[1], bytes[2]]))
}

fn kmeans_palette(points: &[Oklab], requested: usize) -> Vec<PaletteColor> {
    let count = requested.clamp(1, points.len());
    let mut centroids = Vec::with_capacity(count);
    centroids.push(mean_oklab(points));
    while centroids.len() < count {
        let next = points
            .iter()
            .copied()
            .max_by(|left, right| {
                nearest_distance(*left, &centroids).total_cmp(&nearest_distance(*right, &centroids))
            })
            .unwrap_or(centroids[0]);
        centroids.push(next);
    }

    let mut assignments = vec![0_usize; points.len()];
    for _ in 0..12 {
        let mut changed = false;
        for (assignment, point) in assignments.iter_mut().zip(points) {
            let next = nearest_centroid(*point, &centroids);
            changed |= *assignment != next;
            *assignment = next;
        }
        let mut sums = vec![[0.0_f64; 3]; count];
        let mut sizes = vec![0_u32; count];
        for (point, cluster) in points.iter().zip(&assignments) {
            sums[*cluster][0] += f64::from(point.l);
            sums[*cluster][1] += f64::from(point.a);
            sums[*cluster][2] += f64::from(point.b);
            sizes[*cluster] += 1;
        }
        for (index, centroid) in centroids.iter_mut().enumerate() {
            if sizes[index] != 0 {
                let divisor = f64::from(sizes[index]);
                *centroid = Oklab {
                    l: (sums[index][0] / divisor) as f32,
                    a: (sums[index][1] / divisor) as f32,
                    b: (sums[index][2] / divisor) as f32,
                };
            }
        }
        if !changed {
            break;
        }
    }

    let mut sizes = vec![0_usize; count];
    for assignment in assignments {
        sizes[assignment] += 1;
    }
    let mut result: Vec<_> = centroids
        .into_iter()
        .zip(sizes)
        .filter(|(_, size)| *size != 0)
        .map(|(oklab, size)| {
            let rgb = oklab_to_srgb8(oklab);
            PaletteColor {
                oklab,
                proportion: size as f32 / points.len() as f32,
                hex: format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
                name: basic_colour_name(oklab).to_owned(),
            }
        })
        .collect();
    result.sort_by(|left, right| right.proportion.total_cmp(&left.proportion));
    result
}

fn nearest_centroid(point: Oklab, centroids: &[Oklab]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            colour_distance_squared(point, **left)
                .total_cmp(&colour_distance_squared(point, **right))
        })
        .map_or(0, |(index, _)| index)
}

fn nearest_distance(point: Oklab, centroids: &[Oklab]) -> f32 {
    centroids
        .iter()
        .map(|centroid| colour_distance_squared(point, *centroid))
        .min_by(f32::total_cmp)
        .unwrap_or(0.0)
}

fn colour_distance_squared(left: Oklab, right: Oklab) -> f32 {
    (left.l - right.l).powi(2) + (left.a - right.a).powi(2) + (left.b - right.b).powi(2)
}

fn mean_oklab(points: &[Oklab]) -> Oklab {
    let divisor = points.len() as f64;
    let (l, a, b) = points
        .iter()
        .fold((0.0_f64, 0.0_f64, 0.0_f64), |sum, point| {
            (
                sum.0 + f64::from(point.l),
                sum.1 + f64::from(point.a),
                sum.2 + f64::from(point.b),
            )
        });
    Oklab {
        l: (l / divisor) as f32,
        a: (a / divisor) as f32,
        b: (b / divisor) as f32,
    }
}

fn srgb8_to_oklab(rgb: [u8; 3]) -> Oklab {
    let [red, green, blue] = rgb.map(|value| srgb_to_linear(f32::from(value) / 255.0));
    let l = 0.412_221_46 * red + 0.536_332_55 * green + 0.051_445_995 * blue;
    let m = 0.211_903_5 * red + 0.680_699_5 * green + 0.107_396_96 * blue;
    let s = 0.088_302_46 * red + 0.281_718_85 * green + 0.629_978_7 * blue;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    Oklab {
        l: 0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        a: 1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        b: 0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    }
}

fn oklab_to_srgb8(colour: Oklab) -> [u8; 3] {
    let l = colour.l + 0.396_337_78 * colour.a + 0.215_803_76 * colour.b;
    let m = colour.l - 0.105_561_346 * colour.a - 0.063_854_17 * colour.b;
    let s = colour.l - 0.089_484_18 * colour.a - 1.291_485_5 * colour.b;
    let (l, m, s) = (l.powi(3), m.powi(3), s.powi(3));
    let red = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let green = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s;
    let blue = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    [red, green, blue].map(|value| (linear_to_srgb(value).clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn basic_colour_name(colour: Oklab) -> &'static str {
    const SWATCHES: [(&str, [u8; 3]); 12] = [
        ("black", [0, 0, 0]),
        ("white", [255, 255, 255]),
        ("grey", [128, 128, 128]),
        ("red", [220, 40, 40]),
        ("orange", [238, 125, 30]),
        ("yellow", [235, 215, 40]),
        ("green", [40, 170, 70]),
        ("cyan", [35, 190, 195]),
        ("blue", [45, 90, 220]),
        ("purple", [135, 65, 190]),
        ("pink", [230, 100, 160]),
        ("brown", [120, 75, 45]),
    ];
    SWATCHES
        .into_iter()
        .min_by(|(_, left), (_, right)| {
            colour_distance(colour, srgb8_to_oklab(*left))
                .total_cmp(&colour_distance(colour, srgb8_to_oklab(*right)))
        })
        .map_or("unknown", |(name, _)| name)
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn relative_luminance(linear: [f32; 3]) -> f32 {
    0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
}

fn hsv_saturation(rgb: [u8; 3]) -> f32 {
    let [red, green, blue] = rgb.map(|value| f32::from(value) / 255.0);
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    if maximum == 0.0 {
        0.0
    } else {
        (maximum - minimum) / maximum
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().map(|value| f64::from(*value)).sum::<f64>() as f32 / values.len() as f32
}

fn standard_deviation(values: &[f32], mean: f32) -> f32 {
    let variance = values
        .iter()
        .map(|value| f64::from((*value - mean).powi(2)))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() as f32
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    #[test]
    fn categorizes_orientation_and_common_ratios() {
        assert_eq!(
            Orientation::from_dimensions(1920, 1080),
            Orientation::Landscape
        );
        assert_eq!(
            Orientation::from_dimensions(1080, 1920),
            Orientation::Portrait
        );
        assert_eq!(Orientation::from_dimensions(100, 100), Orientation::Square);
        assert_eq!(
            nearest_common_ratio(1920, 1080, 0.03).as_deref(),
            Some("16:9")
        );
        assert_eq!(nearest_common_ratio(1000, 713, 0.01), None);
    }

    #[test]
    fn applies_all_scan_bounds() {
        let bounds = ImportConfig {
            min_width: Some(800),
            max_width: Some(2000),
            min_height: Some(600),
            max_height: Some(1200),
        };
        assert!(within_import_bounds(1920, 1080, &bounds));
        assert!(!within_import_bounds(640, 1080, &bounds));
        assert!(!within_import_bounds(1920, 1440, &bounds));
    }

    #[test]
    fn extracts_deterministic_oklab_palette() {
        let mut image = RgbImage::new(100, 100);
        for (x, _, pixel) in image.enumerate_pixels_mut() {
            *pixel = if x < 75 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        let config = AnalysisConfig {
            palette_colors: 2,
            ..AnalysisConfig::default()
        };
        let first = analyze_pixels(&DynamicImage::ImageRgb8(image.clone()), &config)
            .expect("first analysis");
        let second =
            analyze_pixels(&DynamicImage::ImageRgb8(image), &config).expect("second analysis");
        assert_eq!(first.palette, second.palette);
        assert_eq!(first.palette.len(), 2);
        assert!(first.palette[0].proportion > 0.70);
        assert_eq!(first.palette[0].name, "red");
    }

    #[test]
    fn colour_distance_is_symmetric() {
        let red = parse_hex_colour("#ff0000").expect("red");
        let blue = parse_hex_colour("0000ff").expect("blue");
        assert!((colour_distance(red, blue) - colour_distance(blue, red)).abs() < f32::EPSILON);
        assert_eq!(colour_distance(red, red), 0.0);
    }

    #[test]
    fn threshold_value_is_light() {
        assert_eq!(classify_light_dark(0.5, 0.5), LightDark::Light);
        assert_eq!(classify_light_dark(0.499, 0.5), LightDark::Dark);
    }
}
