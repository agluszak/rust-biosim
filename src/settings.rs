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
    pub rendering_enabled: bool,
    pub old_age_damage_rate: f32,
    pub max_age: u32,
    pub corpse_despawn_delay: u32, // How many turns to wait before despawning dead specimens
    pub hunger_damage_rate: f32,    // Rate at which hunger causes damage
    pub hunger_decrease_rate: f32,  // Rate at which hunger decreases each turn
    pub food_spawn_interval: u32,   // How often food spawns (in turns)
    pub food_restore_amount: f32,   // How much hunger is restored when eating food
    pub max_food_entities: usize,   // Maximum number of food entities in the world
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
                Input::Hunger,        // Add Hunger input
                Input::FoodProximity, // Add FoodProximity input
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
            rendering_enabled: true,
            old_age_damage_rate: 0.1,
            max_age: 500, // Maximum age a specimen can live to before forced death
            corpse_despawn_delay: 30, // Despawn dead specimens after 30 turns
            hunger_damage_rate: 0.5,  // Hunger damage per turn when starving
            hunger_decrease_rate: 0.2, // Hunger decreases by this amount each turn
            food_spawn_interval: 100, // Food spawns every 100 turns
            food_restore_amount: 50.0, // Food restores 50 hunger points
            max_food_entities: 20,    // Maximum of 20 food entities in the world
        }
    }
}
