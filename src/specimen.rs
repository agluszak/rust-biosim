use crate::genome::Genome;
use crate::neural_network::NeuralNetwork;
use bevy::ecs::prelude::*;
use parry2d::na::{Point2, Rotation2};

#[derive(Component)]
pub struct SpecimenData {
    pub oscillator1_period: f32,
    pub oscillator2_period: f32,
    pub oscillator3_period: f32,
    pub distance_traveled: f32,
    pub birthplace: Point2<f32>,
    pub longprobe_distance: f32,
    pub memory1: f32,
    pub memory2: f32,
    pub memory3: f32,
}

impl Default for SpecimenData {
    fn default() -> Self {
        SpecimenData {
            oscillator1_period: 0.0,
            oscillator2_period: 0.0,
            oscillator3_period: 0.0,
            distance_traveled: 0.0,
            birthplace: Point2::new(0.0, 0.0),
            longprobe_distance: 0.0,
            memory1: 0.0,
            memory2: 0.0,
            memory3: 0.0,
        }
    }
}

#[derive(Component)]
pub struct Speed(pub f32);

#[derive(Component)]
pub struct Alive;

pub struct Specimen {
    data: SpecimenData,
    dead: bool,
    genome: Genome,
    brain: NeuralNetwork,
}
