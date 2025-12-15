use nalgebra::DMatrix;
use ndarray::{Array2, Axis};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, random};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use tracing::info;
use tracing_subscriber::FmtSubscriber;

const GENOME_LENGTH: usize = 60; // Adjust as needed
const INPUT_LENGTH: u8 = 100; // Adjust as needed
const OUTPUT_LENGTH: u8 = 100; // Adjust as needed

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Gene {
    weight: i16,
    input_index: u8,
    output_index: u8,
    from_internal: bool,
    to_internal: bool,
}

impl Distribution<Gene> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        Gene::new(
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
        )
    }
}

impl Gene {
    fn new(
        weight: i16,
        input_index: u8,
        output_index: u8,
        from_internal: bool,
        to_internal: bool,
    ) -> Gene {
        Gene {
            weight,
            input_index,
            output_index,
            from_internal,
            to_internal,
        }
    }

    pub fn random() -> Gene {
        rand::rng().random()
    }

    pub fn weight(&self) -> i16 {
        self.weight
    }

    pub fn input_index(&self) -> usize {
        self.input_index as usize
    }

    pub fn output_index(&self) -> usize {
        self.output_index as usize
    }

    pub fn is_from_internal(&self) -> bool {
        self.from_internal
    }

    pub fn is_to_internal(&self) -> bool {
        self.to_internal
    }

    pub fn mutated(&self) -> Gene {
        let mut new_gene = *self;
        let mut rng = rand::rng();
        // Randomly modify one of the fields
        match rng.gen_range(0..5) {
            0 => new_gene.weight = rng.random(),
            1 => new_gene.input_index = rng.random(),
            2 => new_gene.output_index = rng.random(),
            3 => new_gene.from_internal = !new_gene.from_internal,
            _ => new_gene.to_internal = !new_gene.to_internal,
        }
        new_gene
    }

    pub fn crossover(&self, other: &Gene) -> Gene {
        let mut rng = rand::rng();
        Gene {
            weight: if rng.random() {
                self.weight
            } else {
                other.weight
            },
            input_index: if rng.random() {
                self.input_index
            } else {
                other.input_index
            },
            output_index: if rng.random() {
                self.output_index
            } else {
                other.output_index
            },
            from_internal: if rng.random() {
                self.from_internal
            } else {
                other.from_internal
            },
            to_internal: if rng.random() {
                self.to_internal
            } else {
                other.to_internal
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Genome {
    genes: Vec<Gene>,
}

impl Genome {
    pub fn new(genes: Vec<Gene>) -> Genome {
        Genome { genes }
    }

    pub fn random(size: usize) -> Genome {
        let mut genes = Vec::with_capacity(size);
        for _ in 0..size {
            genes.push(Gene::random());
        }
        Genome::new(genes)
    }

    pub fn size(&self) -> usize {
        self.genes.len()
    }

    pub fn genes(&self) -> impl Iterator<Item = &Gene> {
        self.genes.iter()
    }

    pub fn crossover(first: &Genome, second: &Genome) -> Genome {
        let size = first.size();
        assert_eq!(size, second.size());
        Genome::new(
            first
                .genes()
                .zip(second.genes())
                .map(|(f, s)| f.crossover(s))
                .collect(),
        )
    }

    pub fn mutate(&mut self, chance: f32) {
        for gene in self.genes.iter_mut() {
            if random::<f32>() < chance {
                *gene = gene.mutated();
            }
        }
    }
}

fn genome_to_features(genome: &Genome) -> Vec<f32> {
    let mut features = Vec::with_capacity(GENOME_LENGTH * 5);
    features.extend(genome.genes.iter().flat_map(|gene| {
        [
            gene.weight as f32,
            (gene.input_index % INPUT_LENGTH) as f32,
            (gene.output_index % OUTPUT_LENGTH) as f32,
            gene.from_internal as u8 as f32,
            gene.to_internal as u8 as f32,
        ]
    }));
    features
}

fn normalize_features_zscore(features: &[Vec<f32>]) -> Array2<f32> {
    let num_features = features[0].len();
    let num_samples = features.len();

    let mut result = Array2::zeros((num_samples, num_features));

    // Fill result array and calculate means
    let mut means = vec![0.0; num_features];
    for (i, feature) in features.iter().enumerate() {
        for (j, &value) in feature.iter().enumerate() {
            means[j] += value;
            result[[i, j]] = value;
        }
    }

    let num_samples_f = num_samples as f32;
    for mean in &mut means {
        *mean /= num_samples_f;
    }

    // Calculate standard deviations and normalize
    let mut std_devs = vec![0.0; num_features];
    for i in 0..num_samples {
        for j in 0..num_features {
            let diff = result[[i, j]] - means[j];
            std_devs[j] += diff * diff;
        }
    }

    for j in 0..num_features {
        std_devs[j] = (std_devs[j] / (num_samples_f - 1.0)).sqrt().max(1e-10);
        for i in 0..num_samples {
            result[[i, j]] = (result[[i, j]] - means[j]) / std_devs[j];
        }
    }

    result
}

// Optimize PCA by using nalgebra directly and avoiding conversions
fn apply_pca(data: &Array2<f32>, n_components: usize) -> Array2<f32> {
    let n_samples = data.nrows();

    // Calculate mean and center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean.broadcast((n_samples, mean.len())).unwrap();

    // Convert to nalgebra matrix
    let data_matrix = DMatrix::from_row_slice(
        centered.nrows(),
        centered.ncols(),
        centered.as_slice().unwrap(),
    );

    // Compute covariance matrix
    let cov = (&data_matrix.transpose() * &data_matrix) / (n_samples as f32 - 1.0);

    // Compute eigendecomposition
    let eigen = cov.symmetric_eigen();

    // Sort eigenvectors by eigenvalues
    let mut pairs: Vec<_> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    let components = DMatrix::from_columns(
        &pairs
            .iter()
            .take(n_components)
            .map(|(_, v)| *v)
            .collect::<Vec<_>>(),
    );

    // Project data
    let projected = data_matrix * components;

    Array2::from_shape_vec(
        (projected.nrows(), projected.ncols()),
        projected.as_slice().to_vec(),
    )
    .unwrap()
}

fn map_to_rgb(pca_data: &Array2<f32>) -> Vec<(u8, u8, u8)> {
    let mut normalized = pca_data.clone();

    // Normalize each column to [0, 1]
    for mut col in normalized.columns_mut() {
        let min = col.iter().cloned().reduce(f32::min).unwrap_or(0.0);
        let max = col.iter().cloned().reduce(f32::max).unwrap_or(0.0);
        if max - min != 0.0 {
            for val in col.iter_mut() {
                *val = (*val - min) / (max - min);
            }
        } else {
            for val in col.iter_mut() {
                *val = 0.0;
            }
        }
    }

    // Map normalized values to RGB
    normalized
        .rows()
        .into_iter()
        .map(|row| {
            let r = (row[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (row[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (row[2].clamp(0.0, 1.0) * 255.0) as u8;
            (r, g, b)
        })
        .collect()
}

fn assign_colors(genomes: &[Genome]) -> Vec<(u8, u8, u8)> {
    // Convert genomes to feature vectors in parallel
    let feature_vectors: Vec<Vec<f32>> = genomes.par_iter().map(genome_to_features).collect();

    // Normalize features directly to Array2
    let data = normalize_features_zscore(&feature_vectors);

    // Apply PCA to reduce to 3 dimensions
    let pca_data = apply_pca(&data, 3);

    // Map PCA dimensions to RGB
    map_to_rgb(&pca_data)
}

fn main() {
    // Create sample genomes
    let mut genomes = Vec::new();
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("This is an error message");

    info!("Generating Genomes");
    for _ in 0..60 {
        genomes.push(Genome::random(GENOME_LENGTH));
    }

    info!("Assigning Colors");
    let colors = assign_colors(&genomes);

    info!("Printing Colors");

    for (_genome, color) in genomes.iter().zip(colors.iter()) {
        println!("Genome has color RGB{:?}", color);
    }
}
