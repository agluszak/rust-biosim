use std::collections::{HashMap, HashSet};

use num_derive::FromPrimitive;
use variant_count::VariantCount;

use crate::genome::Genome;

type NeuronId = usize;
type Priority = usize;
type Weight = f64;
type Neighbors = HashMap<NeuronId, Vec<NeuronId>>;
type Weights<T> = HashMap<T, Weight>;
type InternalInternal = (NeuronId, NeuronId);
type InternalOutput = (NeuronId, Output);
type InputInternal = (Input, NeuronId);
type InputOutput = (Input, Output);

#[derive(Debug, Clone)]
struct Neuron {
    internal_inputs: Vec<(NeuronId, Weight)>,
    self_weight: f64,
    value: f64,
    previous_value: f64,
}

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    input_internal_weights: Weights<InputInternal>,
    internal_output_weights: Weights<InternalOutput>,
    input_output_weights: Weights<InputOutput>,
    neurons: HashMap<NeuronId, Neuron>,
    computation_order: Vec<NeuronId>,
}

const MAX_WEIGHT: Weight = i16::MAX as Weight;

impl NeuralNetwork {
    pub fn think(&mut self, input_values: &HashMap<Input, f64>) -> HashMap<Output, f64> {
        let mut outputs = HashMap::new();

        for ((input, output), &weight) in self.input_output_weights.iter() {
            let input_value = *input_values.get(input).unwrap_or(&0.0);
            *outputs.entry(*output).or_insert(0.0) += input_value * weight;
        }

        dbg!(&self.input_internal_weights);
        dbg!(&self.internal_output_weights);
        dbg!(&self.neurons);
        for ((input, neuron_id), weight) in self.input_internal_weights.iter() {
            let neuron = self.neurons.get_mut(neuron_id).unwrap();
            neuron.value += input_values.get(input).copied().unwrap_or_default() * *weight;
        }

        for &neuron_id in self.computation_order.iter() {
            let mut neuron = self.neurons[&neuron_id].clone();
            for &(neighbor_id, weight) in &neuron.internal_inputs {
                let neighbor = self.neurons.get(&neighbor_id).unwrap();
                neuron.value += neighbor.value * weight;
            }
            neuron.value += neuron.self_weight * neuron.previous_value;
            neuron.value = neuron.value.tanh();
            neuron.previous_value = neuron.value;
        }

        for ((neuron_id, output), &weight) in self.internal_output_weights.iter() {
            let neuron = self.neurons.get(neuron_id).unwrap();
            *outputs.entry(*output).or_insert(0.0) += neuron.value * weight;
        }

        for (_, value) in outputs.iter_mut() {
            *value = value.tanh();
        }

        outputs
    }

    fn internal_connections(
        internal_internal_weights: &Weights<InternalInternal>,
    ) -> (Neighbors, Neighbors) {
        let mut forward_internal_connections = HashMap::new();
        let mut backward_internal_connections = HashMap::new();

        for (from, to) in internal_internal_weights.keys() {
            backward_internal_connections
                .entry(*to)
                .or_insert(Vec::new())
                .push(*from);
            forward_internal_connections
                .entry(*from)
                .or_insert(Vec::new())
                .push(*to);
        }

        (forward_internal_connections, backward_internal_connections)
    }

    fn topological_sort(
        starting_points: HashSet<NeuronId>,
        neighbors: Neighbors,
    ) -> HashMap<NeuronId, Priority> {
        let mut priorities = HashMap::new();

        for node in starting_points {
            let mut priority = 0;
            let mut visited = HashSet::new();
            let mut stack = Vec::new();
            stack.push(node);
            while let Some(node) = stack.pop() {
                priorities
                    .entry(node)
                    .and_modify(|existing| {
                        if *existing < priority {
                            *existing = priority;
                        }
                    })
                    .or_insert(priority);
                priority += 1;
                visited.insert(node);
                if let Some(neighbors) = neighbors.get(&node) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            stack.push(*neighbor);
                        }
                    }
                }
            }
        }

        priorities
    }

    fn from_weights(
        mut input_internal_weights: Weights<InputInternal>,
        mut internal_output_weights: Weights<InternalOutput>,
        mut internal_internal_weights: Weights<InternalInternal>,
        mut input_output_weights: Weights<InputOutput>,
    ) -> NeuralNetwork {
        // Remove edges that are not connected to any output

        let (forward_internal_connections, backward_internal_connections) =
            NeuralNetwork::internal_connections(&internal_internal_weights);

        let connected_to_inputs = input_internal_weights
            .iter()
            .map(|((_, k), _)| *k)
            .collect::<HashSet<_>>();

        let input_priorities =
            NeuralNetwork::topological_sort(connected_to_inputs, forward_internal_connections);

        let connected_to_outputs = internal_output_weights
            .iter()
            .map(|((k, _), _)| *k)
            .collect::<HashSet<_>>();

        let output_priorities =
            NeuralNetwork::topological_sort(connected_to_outputs, backward_internal_connections);

        internal_output_weights.retain(|(node, _), _| input_priorities.get(node).is_some());
        internal_internal_weights.retain(|(from, to), _| {
            input_priorities.get(from).is_some() && output_priorities.get(to).is_some()
        });
        input_internal_weights.retain(|(_, node), _| output_priorities.get(node).is_some());

        let connected_to_outputs = internal_output_weights
            .iter()
            .map(|((k, _), _)| *k)
            .collect::<HashSet<_>>();

        let (forward_internal_connections, backward_internal_connections) =
            NeuralNetwork::internal_connections(&internal_internal_weights);

        let priorities =
            NeuralNetwork::topological_sort(connected_to_outputs, backward_internal_connections);

        let computation_order = {
            let mut priorities = priorities.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            priorities.sort_unstable_by_key(|(_, v)| *v);
            priorities.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
        };

        let mut neurons = HashMap::new();
        for &neuron_id in priorities.keys() {
            let neuron = Neuron {
                internal_inputs: Vec::new(),
                self_weight: 0.0,
                value: 0.0,
                previous_value: 0.0,
            };
            neurons.insert(neuron_id, neuron);
        }
        for (neuron_id, mut neighbors) in forward_internal_connections {
            let self_pos = neighbors.iter().position(|&neighbor| neighbor == neuron_id);
            let self_weight = if let Some(self_pos) = self_pos {
                neighbors.remove(self_pos);
                *internal_internal_weights
                    .get(&(neuron_id, neuron_id))
                    .unwrap() as f64
                    / MAX_WEIGHT
            } else {
                0.0
            };
            let internal_inputs = neighbors
                .into_iter()
                .map(|neighbor| {
                    let weight = internal_internal_weights
                        .get(&(neuron_id, neighbor))
                        .unwrap();
                    (neighbor, *weight)
                })
                .collect::<Vec<_>>();
            let neuron = neurons.get_mut(&neuron_id).unwrap();
            neuron.internal_inputs = internal_inputs;
            neuron.self_weight = self_weight;
        }

        NeuralNetwork {
            input_internal_weights,
            internal_output_weights,
            input_output_weights,
            neurons,
            computation_order,
        }
    }

    pub fn from_genome(
        genome: &Genome,
        inputs: &[Input],
        outputs: &[Output],
        internal_neurons_count: usize,
    ) -> Self {
        let mut input_internal_weights = HashMap::new();
        let mut internal_output_weights = HashMap::new();
        let mut internal_internal_weights = HashMap::new();
        let mut input_output_weights = HashMap::new();

        for gene in genome.genes() {
            let weight = gene.weight() as isize;
            match (gene.is_from_internal(), gene.is_to_internal()) {
                (false, false) => {
                    let from = inputs[gene.input_index() % inputs.len()];
                    let to = outputs[gene.output_index() % outputs.len()];
                    *input_output_weights.entry((from, to)).or_insert(0.0) +=
                        weight as Weight / MAX_WEIGHT;
                }
                (true, false) => {
                    let from = gene.input_index() % internal_neurons_count;
                    let to = outputs[gene.output_index() % outputs.len()];
                    *internal_output_weights.entry((from, to)).or_insert(0.0) +=
                        weight as Weight / MAX_WEIGHT;
                }
                (false, true) => {
                    let from = inputs[gene.input_index() % inputs.len()];
                    let to = gene.output_index() % internal_neurons_count;
                    *input_internal_weights.entry((from, to)).or_insert(0.0) +=
                        weight as Weight / MAX_WEIGHT;
                }
                (true, true) => {
                    let from = gene.input_index() % internal_neurons_count;
                    let to = gene.output_index() % internal_neurons_count;
                    *internal_internal_weights.entry((from, to)).or_insert(0.0) +=
                        weight as Weight / MAX_WEIGHT;
                }
            }
        }

        NeuralNetwork::from_weights(
            input_internal_weights,
            internal_output_weights,
            internal_internal_weights,
            input_output_weights,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive, VariantCount)]
pub enum Input {
    PosX,
    PosY,
    Direction,
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
    Memory1,
    Memory2,
    Memory3,
    // Energy,
    // Health,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive, VariantCount)]
pub enum Output {
    // Actions
    Move,
    Turn,
    Remember1,
    Remember2,
    Remember3,
    ChangeSpeed,
    // Analog values
    SetSpeed,
    SetDirection,
    SetMemory1,
    SetMemory2,
    SetMemory3,
    SetOscillator1Period,
    SetOscillator2Period,
    SetOscillator3Period,
    SetLongProbeDistance,
}
