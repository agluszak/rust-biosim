mod genome;
mod neural_network;
mod settings;
mod specimen;

use bevy::prelude::*;
use bevy_prototype_lyon::entity::ShapeBundle;
use bevy_prototype_lyon::prelude::*;
use parry2d::na::distance;
use rand::random;
use std::collections::HashMap;

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(ShapePlugin)
        .add_startup_system(setup_system)
        .add_system(display_system)
        .add_system(movement_system)
        .add_system(brain_input_collection_system)
        .add_system(thinking_system)
        .add_system(doing_system)
        .run();
}

#[derive(Component)]
struct Speed(f32);

#[derive(Component)]
struct Size(f32);

#[derive(Component)]
struct Position(parry2d::na::Point2<f32>);

#[derive(Component)]
struct Direction(parry2d::na::Rotation2<f32>);

#[derive(Component)]
struct Age(u32);

#[derive(Component)]
struct Birthplace(parry2d::na::Point2<f32>);

#[derive(Component)]
struct Memory([f64; 3]);

#[derive(Component)]
struct Oscillator([f64; 3]);

#[derive(Component)]
struct Genome(genome::Genome);

#[derive(Component)]
struct Brain(neural_network::NeuralNetwork);

#[derive(Component)]
struct Alive;

#[derive(Component, Default)]
struct BrainInputs(HashMap<neural_network::Input, f64>);

impl BrainInputs {
    fn add(&mut self, input: neural_network::Input, value: f64) {
        assert!(value >= -1.0 && value <= 1.0);
        self.0.insert(input, value);
    }

    fn read(&self) -> &HashMap<neural_network::Input, f64> {
        &self.0
    }
}

#[derive(Component, Default)]
struct BrainOutputs(HashMap<neural_network::Output, f64>);

impl BrainOutputs {
    fn set(&mut self, outputs: HashMap<neural_network::Output, f64>) {
        self.0 = outputs;
    }

    fn get(&self, output: neural_network::Output) -> f64 {
        self.0.get(&output).copied().unwrap_or(0.0)
    }

    fn activated(&self, output: neural_network::Output) -> bool {
        self.get(output) > 0.5 // TODO make this configurable
    }
}

#[derive(Bundle)]
struct SpecimenBundle {
    speed: Speed,
    position: Position,
    direction: Direction,
    birthplace: Birthplace,
    memory: Memory,
    oscillator: Oscillator,
    genome: Genome,
    brain: Brain,
    brain_inputs: BrainInputs,
    brain_outputs: BrainOutputs,
    alive: Alive,
    #[bundle]
    shape_bundle: ShapeBundle,
}

#[derive(Component)]
struct WantsToMove;

fn map_range(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    assert!(from_min < from_max);
    assert!(to_min < to_max);
    assert!(value >= from_min && value <= from_max);
    let value = (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min;
    assert!(value >= to_min && value <= to_max);
    value
}

fn normalize(value: f64, min: f64, max: f64) -> f64 {
    map_range(value, min, max, -1.0, 1.0)
}

fn brain_input_collection_system(
    mut query: Query<(
        &mut BrainInputs,
        &Position,
        &Speed,
        &Direction,
        &Birthplace,
        With<Alive>,
    )>,
) {
    for (mut brain_inputs, position, speed, direction, birthplace, _) in &mut query.iter_mut() {
        brain_inputs.add(
            neural_network::Input::PosX,
            normalize(position.0.x as f64, -50.0, 50.0),
        ); // TODO: use a res for world size
        brain_inputs.add(
            neural_network::Input::PosY,
            normalize(position.0.y as f64, -50.0, 50.0),
        );
        brain_inputs.add(
            neural_network::Input::Speed,
            normalize(speed.0 as f64, 0.5, 2.0),
        );
        brain_inputs.add(
            neural_network::Input::Direction,
            direction.0.angle().sin() as f64,
        );
        brain_inputs.add(
            neural_network::Input::DistanceToBirthplace,
            normalize(
                distance(&birthplace.0, &position.0) as f64,
                0.0,
                20000.0f64.sqrt(),
            ),
            // TODO: use a res for world size
        );
        brain_inputs.add(neural_network::Input::Random, normalize(random(), 0.0, 1.0));
    }
}

fn thinking_system(mut query: Query<(&mut Brain, &BrainInputs, &mut BrainOutputs)>) {
    for (mut brain, brain_inputs, mut brain_outputs) in &mut query.iter_mut() {
        let outputs = brain.0.think(brain_inputs.read());
        brain_outputs.set(outputs);
    }
}

fn doing_system(
    mut commands: Commands,
    mut query: Query<(Entity, &BrainOutputs, &mut Direction, &mut Speed)>,
) {
    let mut done = false;
    for (entity, brain_outputs, mut direction, mut speed) in &mut query.iter_mut() {
        if !done {
            done = true;
            let current_direction = direction.0.angle();
            let current_speed = speed.0;
            let desired_speed = brain_outputs.get(neural_network::Output::ChangeSpeed);
            let desired_speed = map_range(desired_speed, -1.0, 1.0, 0.5, 2.0);
            let desired_direction = brain_outputs.get(neural_network::Output::SetDirection);
            let desired_direction =
                parry2d::na::Rotation2::new(desired_direction.asin() as f32).angle();
            dbg!(
                entity,
                current_direction,
                current_speed,
                desired_speed,
                desired_direction
            );
        }
        if brain_outputs.activated(neural_network::Output::Move) {
            commands.entity(entity).insert(WantsToMove);
        }
        if brain_outputs.activated(neural_network::Output::Turn) {
            let output = brain_outputs.get(neural_network::Output::SetDirection);
            direction.0 = parry2d::na::Rotation2::new(output.asin() as f32);
        }
        if brain_outputs.activated(neural_network::Output::ChangeSpeed) {
            let output = brain_outputs.get(neural_network::Output::ChangeSpeed);
            speed.0 = map_range(output, -1.0, 1.0, 0.5, 2.0) as f32;
        }
    }
}

fn movement_system(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Position, &Speed, &Direction, With<WantsToMove>)>,
) {
    for (entity, mut position, speed, direction, _) in query.iter_mut() {
        position.0 += direction.0 * parry2d::na::Vector2::new(speed.0 * 10.0, 0.0);
        position.0.x = position.0.x.max(-50.0).min(50.0);
        position.0.y = position.0.y.max(-50.0).min(50.0);
        commands.entity(entity).remove::<WantsToMove>();
        // TODO: use a res for speed and world size
    }
}

fn display_system(mut query: Query<(&Position, &mut Transform)>) {
    for (position, mut transform) in query.iter_mut() {
        transform.translation = Vec3::new(position.0.x * 5.0, position.0.y * 5.0, 0.0);
    }
}

impl SpecimenBundle {
    // 100
    fn new(
        world_size: f32,
        genome_length: usize,
        inputs: &[neural_network::Input],
        outputs: &[neural_network::Output],
        internal_neurons: usize,
    ) -> Self {
        let speed = Speed(1.0); // TODO this is a speed multiplier
        let position = Position(parry2d::na::Point2::new(
            rand::random::<f32>() * world_size - world_size / 2.0,
            rand::random::<f32>() * world_size - world_size / 2.0,
        ));
        let birthplace = Birthplace(position.0);
        let direction = Direction(parry2d::na::Rotation2::new(
            random::<f32>() * 2.0 * std::f32::consts::PI,
        ));

        let shape = shapes::Circle {
            radius: 10.0,
            center: Vec2::new(0.0, 0.0),
        };

        let shape_bundle = GeometryBuilder::build_as(
            &shape,
            DrawMode::Outlined {
                fill_mode: FillMode::color(Color::rgb(random(), random(), random())),
                outline_mode: StrokeMode::new(Color::BLACK, 1.0),
            },
            // TODO world scaling
            Transform::from_translation(Vec3::new(position.0.x * 10.0, position.0.y * 10.0, 0.0)),
        );

        let genome = Genome(genome::Genome::random(genome_length));
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
            direction,
            birthplace,
            memory: Memory([0.0; 3]),
            oscillator: Oscillator([0.0; 3]),
            genome,
            brain,
            brain_inputs,
            brain_outputs,
            alive: Alive,
            shape_bundle,
        }
    }
}

fn setup_system(mut commands: Commands) {
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());

    // TODO
    let inputs = {
        use neural_network::Input;
        [
            Input::PosX,
            Input::PosY,
            Input::Direction,
            Input::Speed,
            // Input::Age,
            Input::Random,
            Input::DistanceToBirthplace,
        ]
    };

    let outputs = {
        use neural_network::Output;
        [
            Output::Move,
            Output::Turn,
            Output::ChangeSpeed,
            Output::SetSpeed,
            Output::SetDirection,
        ]
    };

    let internal_neurons = 10;

    let genome_length = 20;

    for _ in 0..100 {
        commands.spawn_bundle(SpecimenBundle::new(
            100.0,
            genome_length,
            &inputs,
            &outputs,
            internal_neurons,
        ));
    }
}
