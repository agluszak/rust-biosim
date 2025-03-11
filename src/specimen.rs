use crate::{MEMORY_SIZE, genome, map_range, neural_network};
use bevy::prelude::*;
use bevy_prototype_lyon::entity::ShapeBundle;
use bevy_prototype_lyon::prelude::*;
use rand::random;
use std::collections::HashMap;

#[derive(Component)]
pub struct Health(pub f32);

#[derive(Component, Debug)]
pub struct Hunger(pub f32);

#[derive(Component, Debug)]
pub struct SpeedMultiplier(pub f32);

impl SpeedMultiplier {
    const MAX: f32 = 2.0;
}

impl Default for SpeedMultiplier {
    fn default() -> Self {
        SpeedMultiplier(1.0)
    }
}

impl NeuronValueConvertible for SpeedMultiplier {
    fn set_from_neuron_value(&mut self, neuron_value: &NeuronValue) {
        self.0 = neuron_value.as_multiplier(Self::MAX);
    }

    fn get_neuron_value(&self) -> NeuronValue {
        NeuronValue::from_multiplier(self.0, Self::MAX)
    }
}

#[derive(Component, Debug)]
pub struct Size(pub f32);

#[derive(Component, Debug)]
pub struct Position(pub parry2d::na::Point2<f32>);

#[derive(Component, Debug)]
pub struct Direction(pub parry2d::na::Rotation2<f32>);

impl Default for Direction {
    fn default() -> Self {
        Direction(parry2d::na::Rotation2::new(0.0))
    }
}

impl Direction {
    pub fn x(&self) -> NeuronValue {
        NeuronValue::new(self.0.angle().cos())
    }

    pub fn y(&self) -> NeuronValue {
        NeuronValue::new(self.0.angle().sin())
    }
}

#[derive(Component, Debug)]
pub struct PreviousPosition(pub parry2d::na::Point2<f32>);

#[derive(Component, Debug)]
pub struct Age(pub u32);

#[derive(Component, Debug)]
pub struct Birthplace(pub parry2d::na::Point2<f32>);

#[derive(Component)]
pub struct Memory(pub [f32; MEMORY_SIZE]);

#[derive(Component)]
pub struct Oscillator(pub [f32; 3]);

#[derive(Component)]
pub struct Genome(pub genome::Genome);

#[derive(Component)]
pub struct Brain(pub neural_network::NeuralNetwork);

#[derive(Component)]
pub struct Alive;

#[derive(Component)]
pub struct DeathTurn(pub u32); // Store the turn number when the specimen died

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NeuronInput(f32);

#[derive(Component, Default)]
pub struct BrainInputs(HashMap<neural_network::Input, f32>);

impl BrainInputs {
    pub fn add(&mut self, input: neural_network::Input, value: NeuronValue) {
        self.0.insert(input, value.value());
    }

    pub fn read(&self) -> &HashMap<neural_network::Input, f32> {
        &self.0
    }
}

pub trait NeuronValueConvertible {
    fn set_from_neuron_value(&mut self, neuron_value: &NeuronValue);
    fn get_neuron_value(&self) -> NeuronValue;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NeuronValue(f32);

impl NeuronValue {
    const MIN: f32 = -1.0;
    const MAX: f32 = 1.0;

    pub fn new(value: f32) -> Self {
        assert!((Self::MIN..=Self::MAX).contains(&value));
        NeuronValue(value)
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    pub fn as_linear(&self, min: f32, max: f32) -> f32 {
        map_range(self.0, Self::MIN, Self::MAX, min, max)
    }

    pub fn from_linear(value: f32, min: f32, max: f32) -> Self {
        Self::new(map_range(value, min, max, Self::MIN, Self::MAX))
    }

    pub fn as_multiplier(&self, max: f32) -> f32 {
        max.powf(self.0)
    }

    pub fn from_multiplier(multiplier: f32, max: f32) -> Self {
        NeuronValue::new(multiplier.log(max))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn neuron_value_from_to_multiplier(neuron_value: f32, max: f32) {
        let from_value = NeuronValue::new(neuron_value);
        let multiplier = from_value.as_multiplier(max);
        let actual = NeuronValue::from_multiplier(multiplier, max);
        assert!((actual.value() - neuron_value).abs() < 0.00001);
    }

    #[test]
    fn test_neuron_value_from_to_multiplier() {
        neuron_value_from_to_multiplier(0.0, 2.0);
        neuron_value_from_to_multiplier(1.0, 2.0);
        neuron_value_from_to_multiplier(-1.0, 2.0);
        neuron_value_from_to_multiplier(0.33, 2.0);
        neuron_value_from_to_multiplier(0.5, 2.0);
        neuron_value_from_to_multiplier(0.66, 2.0);
        neuron_value_from_to_multiplier(-0.33, 2.0);
        neuron_value_from_to_multiplier(-0.5, 2.0);
        neuron_value_from_to_multiplier(-0.66, 2.0);
    }
}

#[derive(Component, Default)]
pub struct BrainOutputs(HashMap<neural_network::Output, f32>);

impl BrainOutputs {
    pub fn from_hashmap(outputs: HashMap<neural_network::Output, f32>) -> Self {
        BrainOutputs(outputs)
    }

    pub fn get(&self, output: neural_network::Output) -> NeuronValue {
        NeuronValue::new(self.0.get(&output).copied().unwrap_or(0.0))
    }

    pub fn activated(&self, output: neural_network::Output) -> bool {
        self.get(output).value() > 0.0 // TODO make this configurable
    }
}

#[derive(Bundle)]
pub struct SpecimenBundle {
    speed: SpeedMultiplier,
    position: Position,
    previous_position: PreviousPosition,
    direction: Direction,
    birthplace: Birthplace,
    memory: Memory,
    oscillator: Oscillator,
    genome: Genome,
    brain: Brain,
    brain_inputs: BrainInputs,
    brain_outputs: BrainOutputs,
    health: Health,
    hunger: Hunger, // Add hunger component
    alive: Alive,
    size: Size,
    age: Age,
    #[bundle()]
    shape_bundle: ShapeBundle,
    fill: Fill,
    stroke: Stroke,
}

impl SpecimenBundle {
    pub fn new(
        genome: Genome,
        inputs: &[neural_network::Input],
        outputs: &[neural_network::Output],
        internal_neurons: usize,
        position: Position,
    ) -> Self {
        let speed = SpeedMultiplier::default(); // TODO this is a speed multiplier
        let previous_position = PreviousPosition(position.0);
        let birthplace = Birthplace(position.0);
        let direction = Direction(parry2d::na::Rotation2::new(
            random::<f32>() * 2.0 * std::f32::consts::PI,
        ));

        // Set initial size
        let size = Size(10.0);

        let shape = shapes::Circle {
            radius: size.0,
            center: Vec2::new(0.0, 0.0),
        };
        let path = GeometryBuilder::build_as(&shape);
        let shape_bundle = ShapeBundle { path, ..default() };
        let brain = Brain(neural_network::NeuralNetwork::from_genome(
            &genome.0,
            inputs,
            outputs,
            internal_neurons,
        ));
        let brain_inputs = BrainInputs::default();
        let brain_outputs = BrainOutputs::default();

        SpecimenBundle {
            speed,
            position,
            previous_position,
            direction,
            birthplace,
            memory: Memory([0.0; MEMORY_SIZE]),
            oscillator: Oscillator([1.0; 3]),
            genome,
            brain,
            brain_inputs,
            brain_outputs,
            health: Health(100.0),
            hunger: Hunger(100.0), // Start with full belly
            alive: Alive,
            size,
            age: Age(0), // Starting age is 0
            shape_bundle,
            fill: Fill::color(Color::srgb(random(), random(), random())),
            stroke: Stroke::new(Color::BLACK, 1.0),
        }
    }
}

// Add a component to mark food entities
#[derive(Component)]
pub struct Food;
