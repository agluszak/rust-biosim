use crate::neural_network;
use bevy::ecs::prelude::*;

#[derive(Resource, Clone, Debug)]
pub struct Settings {
    pub specimen_size: f32,
    pub world_size: f32,
    pub world_half_size: f32,
    pub population: usize,
    pub genome_length: usize,
    pub internal_neurons: usize,
    pub mutation_chance: f32,
    pub turns_per_generation: u32,
    pub proximity_distance: f32,
    pub default_longprobe_distance: f32,
    pub fast_oscillator_frequency: f32,
    pub medium_oscillator_frequency: f32,
    pub slow_oscillator_frequency: f32,
    pub base_speed: f32,
    pub brain_inputs: Vec<neural_network::Input>,
    pub brain_outputs: Vec<neural_network::Output>,
}

pub const MEMORY_SIZE: usize = 4;

impl Default for Settings {
    fn default() -> Settings {
        let brain_inputs = {
            use neural_network::Input;
            let mut brain_inputs = vec![
                Input::PosX,
                Input::PosY,
                Input::DirectionX,
                Input::DirectionY,
                Input::Speed,
                Input::Age,
                Input::Random,
                Input::DistanceToBirthplace,
                Input::DistanceTravelled,
            ];
            brain_inputs.extend((0..MEMORY_SIZE).map(Input::Memory));
            brain_inputs
        };

        let brain_outputs = {
            use neural_network::Output;
            let mut brain_outputs = vec![
                Output::Move,
                Output::Turn,
                Output::ChangeSpeed,
                // Analog
                Output::DesiredSpeed,
                Output::DesiredDirectionX,
                Output::DesiredDirectionY,
            ];
            brain_outputs.extend(
                (0..MEMORY_SIZE).flat_map(|i| vec![Output::DesiredMemory(i), Output::Remember(i)]),
            );
            brain_outputs
        };

        let world_size = 100.0;

        Settings {
            specimen_size: 1.0,
            world_size,
            world_half_size: world_size / 2.0,
            population: 150,
            internal_neurons: 30,
            genome_length: 70,
            mutation_chance: 0.01,
            turns_per_generation: 150,
            proximity_distance: 3.0,
            default_longprobe_distance: 10.0,
            fast_oscillator_frequency: 0.3,
            medium_oscillator_frequency: 0.09,
            slow_oscillator_frequency: 0.027,
            base_speed: 1.0,
            brain_inputs,
            brain_outputs,
        }
    }
}
