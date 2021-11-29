use crate::genome::Genome;
use crate::neural_network::NeuralNetwork;
use cgmath::{Point2, Rad};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecimenData {
    pub position: Point2<f32>,
    pub rotation: Rad<f32>,
    pub move_successful: bool,
    pub oscillator1_period: f32,
    pub oscillator2_period: f32,
    pub oscillator3_period: f32,
    pub distance_traveled: f32,
    pub birthplace: Point2<f32>,
    pub longprobe_distance: f32,
    pub memory1: f32,
    pub memory2: f32,
    pub memory3: f32,
    pub speed: f32,
}

impl Default for SpecimenData {
    fn default() -> Self {
        SpecimenData {
            position: Point2::new(0.0, 0.0),
            rotation: Rad(0.0),
            move_successful: false,
            oscillator1_period: 0.0,
            oscillator2_period: 0.0,
            oscillator3_period: 0.0,
            distance_traveled: 0.0,
            birthplace: Point2::new(0.0, 0.0),
            longprobe_distance: 0.0,
            memory1: 0.0,
            memory2: 0.0,
            memory3: 0.0,
            speed: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Specimen {
    data: SpecimenData,
    dead: bool,
    genome: Genome,
    brain: NeuralNetwork,
}

impl Specimen {
    fn new(data: SpecimenData, genome: Genome) -> Self {
        Specimen {
            data,
            dead: false,
            genome,
            brain: NeuralNetwork {}// genome.network(),
        }
    }
}
