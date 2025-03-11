use std::collections::{HashMap, HashSet};

use variant_count::VariantCount;

use crate::genome::Genome;

type Weight = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, VariantCount)]
pub enum Input {
    PosX,
    PosY,
    DirectionX,
    DirectionY,
    Speed,
    Age,
    Proximity,
    LongProbe,
    Random,
    DistanceTravelled,
    DistanceToBirthplace,
    LongProbeDistance,
    Oscillator1, // PI * time * 5 / max_time
    Oscillator2, // PI * time * 25 / max_time
    Oscillator3, // PI * time * 125 / max_time
    Memory(usize),
    Hunger,       // Add hunger input
    FoodProximity, // Add food proximity input
    // Energy,
    // Health,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, VariantCount)]
pub enum Output {
    // Actions
    Move,
    Turn,
    Remember(usize),
    ChangeSpeed,
    // Analog values
    DesiredSpeed,
    DesiredDirectionX,
    DesiredDirectionY,
    DesiredMemory(usize),
    DesiredFastOscillatorPeriod,
    DesiredMediumOscillatorPeriod,
    DesiredSlowOscillatorPeriod,
    DesiredLongProbeDistance,
}

const MAX_WEIGHT: f32 = i16::MAX as f32;

/// High-performance neural network implementation using flat arrays and direct indexing.
/// This is much more cache-friendly than the HashMap-based approach.
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    // New optimized structure using continuous memory
    num_internal_neurons: usize,
    input_indices: Vec<Input>,   // Maps from index to Input enum
    output_indices: Vec<Output>, // Maps from index to Output enum

    // Input-to-internal connectivity
    input_to_internal: Vec<(usize, usize, f32)>, // (input_idx, neuron_idx, weight)

    // Internal-to-output connectivity
    internal_to_output: Vec<(usize, usize, f32)>, // (neuron_idx, output_idx, weight)

    // Direct input-to-output connectivity
    input_to_output: Vec<(usize, usize, f32)>, // (input_idx, output_idx, weight)

    // Internal neuron connectivity
    internal_weights: Vec<Vec<(usize, f32)>>, // For each neuron, list of (source_idx, weight)
    self_weights: Vec<f32>,                   // Self-recurrent weights

    // Computation state
    neuron_values: Vec<f32>,   // Current values
    neuron_previous: Vec<f32>, // Previous tick values

    // Computation order (in topological order)
    computation_order: Vec<usize>,
}

impl NeuralNetwork {
    /// Optimized neural network inference that processes inputs and produces outputs
    pub fn think(&mut self, input_values: &HashMap<Input, f32>) -> HashMap<Output, f32> {
        // Initialize outputs with zeros
        let mut output_values = vec![0.0; self.output_indices.len()];

        // Process direct input-to-output connections (no activation function)
        for &(input_idx, output_idx, weight) in &self.input_to_output {
            let input_enum = self.input_indices[input_idx];
            let input_value = input_values.get(&input_enum).copied().unwrap_or(0.0);
            output_values[output_idx] += input_value * weight;
        }

        // Reset all neuron values to zero
        for value in &mut self.neuron_values {
            *value = 0.0;
        }

        // Process inputs to internal neurons
        for &(input_idx, neuron_idx, weight) in &self.input_to_internal {
            let input_enum = self.input_indices[input_idx];
            let input_value = input_values.get(&input_enum).copied().unwrap_or(0.0);
            self.neuron_values[neuron_idx] += input_value * weight;
        }

        // Process internal neurons in topological order
        for &neuron_idx in &self.computation_order {
            // Apply self-recurrent connection
            self.neuron_values[neuron_idx] +=
                self.self_weights[neuron_idx] * self.neuron_previous[neuron_idx];

            // Calculate value based on inputs from other neurons
            for &(src_idx, weight) in &self.internal_weights[neuron_idx] {
                self.neuron_values[neuron_idx] += self.neuron_values[src_idx] * weight;
            }

            // Apply activation function
            self.neuron_values[neuron_idx] = self.neuron_values[neuron_idx].tanh();

            // Store for next iteration
            self.neuron_previous[neuron_idx] = self.neuron_values[neuron_idx];
        }

        // Process internal-to-output connections
        for &(neuron_idx, output_idx, weight) in &self.internal_to_output {
            output_values[output_idx] += self.neuron_values[neuron_idx] * weight;
        }

        // Apply activation function to outputs
        for value in &mut output_values {
            *value = value.tanh();
        }

        // Convert to HashMap for API compatibility
        let mut outputs = HashMap::with_capacity(self.output_indices.len());
        for (idx, &output_enum) in self.output_indices.iter().enumerate() {
            outputs.insert(output_enum, output_values[idx]);
        }

        outputs
    }

    // Create a new neural network from a genome
    pub fn from_genome(
        genome: &Genome,
        inputs: &[Input],
        outputs: &[Output],
        internal_neurons_count: usize,
    ) -> Self {
        // Create mappings between enum values and indices
        let input_indices: Vec<Input> = inputs.to_vec();
        let output_indices: Vec<Output> = outputs.to_vec();

        // Create a temporary representation using HashMaps
        let mut input_to_internal: Vec<(usize, usize, f32)> = Vec::new();
        let mut internal_to_output: Vec<(usize, usize, f32)> = Vec::new();
        let mut input_to_output: Vec<(usize, usize, f32)> = Vec::new();

        // HashMap for accumulating weights during genome processing
        let mut io_weights: HashMap<(usize, usize), f32> = HashMap::new();
        let mut ii_weights: HashMap<(usize, usize), f32> = HashMap::new();
        let mut oi_weights: HashMap<(usize, usize), f32> = HashMap::new();

        // Process genes and accumulate weights
        for gene in genome.genes() {
            let weight = gene.weight() as isize as f32 / MAX_WEIGHT;

            match (gene.is_from_internal(), gene.is_to_internal()) {
                (false, false) => {
                    // Input to output
                    let input_idx = gene.input_index() % inputs.len();
                    let output_idx = gene.output_index() % outputs.len();
                    *io_weights.entry((input_idx, output_idx)).or_insert(0.0) += weight;
                }
                (true, false) => {
                    // Internal to output
                    let neuron_idx = gene.input_index() % internal_neurons_count;
                    let output_idx = gene.output_index() % outputs.len();
                    *oi_weights.entry((neuron_idx, output_idx)).or_insert(0.0) += weight;
                }
                (false, true) => {
                    // Input to internal
                    let input_idx = gene.input_index() % inputs.len();
                    let neuron_idx = gene.output_index() % internal_neurons_count;
                    *ii_weights.entry((input_idx, neuron_idx)).or_insert(0.0) += weight;
                }
                (true, true) => {
                    // Internal to internal
                    let from_idx = gene.input_index() % internal_neurons_count;
                    let to_idx = gene.output_index() % internal_neurons_count;
                    // Handled separately below
                }
            }
        }

        // Convert accumulated weights to connection vectors
        for ((input_idx, output_idx), weight) in io_weights {
            input_to_output.push((input_idx, output_idx, weight));
        }

        for ((input_idx, neuron_idx), weight) in ii_weights {
            input_to_internal.push((input_idx, neuron_idx, weight));
        }

        for ((neuron_idx, output_idx), weight) in oi_weights {
            internal_to_output.push((neuron_idx, output_idx, weight));
        }

        // Create the adjacency list of the internal network
        let mut adjacency_list: Vec<Vec<(usize, f32)>> = vec![Vec::new(); internal_neurons_count];
        let mut self_weights: Vec<f32> = vec![0.0; internal_neurons_count];

        // Process internal-to-internal connections
        for gene in genome.genes() {
            if gene.is_from_internal() && gene.is_to_internal() {
                let from_idx = gene.input_index() % internal_neurons_count;
                let to_idx = gene.output_index() % internal_neurons_count;
                let weight = gene.weight() as isize as f32 / MAX_WEIGHT;

                if from_idx == to_idx {
                    // Self-connection
                    self_weights[from_idx] += weight;
                } else {
                    // Connection to another neuron
                    adjacency_list[to_idx].push((from_idx, weight));
                }
            }
        }

        // Calculate the topological order for execution
        let computation_order =
            Self::compute_topological_order(&adjacency_list, internal_neurons_count);

        // Create the neural network
        NeuralNetwork {
            num_internal_neurons: internal_neurons_count,
            input_indices,
            output_indices,
            input_to_internal,
            internal_to_output,
            input_to_output,
            internal_weights: adjacency_list,
            self_weights,
            neuron_values: vec![0.0; internal_neurons_count],
            neuron_previous: vec![0.0; internal_neurons_count],
            computation_order,
        }
    }

    /// Compute an efficient topological ordering of neurons for forward propagation
    fn compute_topological_order(
        adjacency_list: &[Vec<(usize, f32)>],
        num_neurons: usize,
    ) -> Vec<usize> {
        // A simple implementation of Kahn's algorithm for topological sorting
        let mut in_degree = vec![0; num_neurons];
        let mut result = Vec::with_capacity(num_neurons);

        // Calculate in-degrees
        for connections in adjacency_list {
            for &(src, _) in connections {
                in_degree[src] += 1;
            }
        }

        // Start with nodes that have no incoming edges
        let mut queue: Vec<usize> = (0..num_neurons).filter(|&i| in_degree[i] == 0).collect();

        while let Some(node) = queue.pop() {
            result.push(node);

            // Reduce in-degree for all neighbors
            for &(neighbor, _) in &adjacency_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push(neighbor);
                }
            }
        }

        // If we have a cycle, just return nodes in their order (could be improved)
        if result.len() < num_neurons {
            for i in 0..num_neurons {
                if !result.contains(&i) {
                    result.push(i);
                }
            }
        }

        result
    }
}
