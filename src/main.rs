mod genome;
mod neural_network;
mod settings;
mod specimen;

use crate::settings::{MEMORY_SIZE, Settings};
use crate::specimen::{
    Age, Alive, Birthplace, Brain, BrainInputs, BrainOutputs, Direction, Genome, Health, Memory,
    NeuronValue, NeuronValueConvertible, Oscillator, Position, PreviousPosition, SpecimenBundle,
    SpeedMultiplier,
};
use bevy::DefaultPlugins;
use bevy::app::{App, Startup, Update};
use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy_prototype_lyon::prelude::*;
use parry2d::na::{Rotation2, Vector2, distance};
use rand::prelude::IndexedRandom;
use rand::random;
use std::time::Instant;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Turbo Evolution Giga Simulator".to_string(),
                resolution: (550., 550.).into(),
                present_mode: PresentMode::Immediate,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(ShapePlugin)
        .add_systems(Startup, setup_system)
        .add_systems(Update, display_system)
        .add_systems(Update, movement_system)
        .add_systems(Update, brain_input_collection_system)
        .add_systems(Update, thinking_system)
        .add_systems(Update, doing_system)
        .add_systems(Update, time_system)
        .add_systems(Update, text_update_system)
        .add_systems(Update, transparency_system)
        .add_systems(
            Update,
            (
                first_generation_system.run_if(is_first_turn),
                damage_system,
                death_system,
                new_generation_system,
            )
                .chain(),
        )
        .run();
}

#[derive(Component)]
struct WantsToMove;

#[derive(Resource)]
struct Turn(u32);

#[derive(Resource)]
struct Generation(u32);

#[derive(Resource)]
struct GenerationStartTime(std::time::Instant);

#[derive(Component)]
struct TurnText;

#[inline]
fn map_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    assert!(from_min <= from_max);
    assert!(to_min <= to_max);
    assert!(value >= from_min && value <= from_max);
    let value = (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min;
    assert!(value >= to_min && value <= to_max);
    value
}

fn brain_input_collection_system(
    mut query: Query<
        (
            &mut BrainInputs,
            &Position,
            &SpeedMultiplier,
            &Direction,
            &Birthplace,
            &Oscillator,
            &PreviousPosition,
            &Memory,
        ),
        With<Alive>,
    >,
    turn: Res<Turn>,
    settings: Res<Settings>,
) {
    use neural_network::Input;
    for (
        mut brain_inputs,
        position,
        speed,
        direction,
        birthplace,
        _oscillator,
        previous_position,
        memory,
    ) in &mut query.iter_mut()
    {
        brain_inputs.add(
            Input::PosX,
            NeuronValue::from_linear(
                position.0.x,
                -settings.world_half_size,
                settings.world_half_size,
            ),
        );
        brain_inputs.add(
            Input::PosY,
            NeuronValue::from_linear(
                position.0.y,
                -settings.world_half_size,
                settings.world_half_size,
            ),
        );
        brain_inputs.add(Input::Speed, speed.get_neuron_value());
        brain_inputs.add(Input::DirectionX, direction.x());
        brain_inputs.add(Input::DirectionY, direction.y());
        brain_inputs.add(
            Input::DistanceToBirthplace,
            NeuronValue::from_linear(
                distance(&birthplace.0, &position.0),
                0.0,
                settings.world_size * std::f32::consts::SQRT_2,
            ),
        );
        brain_inputs.add(Input::Random, NeuronValue::from_linear(random(), 0.0, 1.0));
        brain_inputs.add(
            Input::DistanceTravelled,
            NeuronValue::from_linear(
                distance(&position.0, &previous_position.0),
                0.0,
                settings.base_speed * 2.0, // TODO use actual speed (need to use previous value)
            ),
        );
        brain_inputs.add(
            Input::Age,
            NeuronValue::from_linear(
                turn.0 as f32 / settings.turns_per_generation as f32,
                // TODO use age
                0.0,
                1.0,
            ),
        );
        for i in 0..MEMORY_SIZE {
            brain_inputs.add(Input::Memory(i), NeuronValue::new(memory.0[i]));
        }
    }
}

fn new_generation_system(
    mut commands: Commands,
    settings: Res<Settings>,
    generation: Res<Generation>,
    turn: Res<Turn>,
    to_remove: Query<Entity, With<Genome>>,
    alive: Query<(&Genome, &Position), With<Alive>>,
) {
    if turn.0 == settings.turns_per_generation {
        // Get genomes from surviving specimens
        let genomes = alive
            .iter()
            .map(|(genome, _)| genome.0.clone())
            .collect::<Vec<_>>();

        println!("Generation {}: {}", generation.0, genomes.len());

        // Remove all existing specimens
        for entity in to_remove.iter() {
            commands.entity(entity).despawn();
        }

        // If no specimens survived, panic
        if genomes.is_empty() {
            panic!("No specimens survived!");
        }

        // Repopulate using the surviving specimens' genomes
        for _ in 0..settings.population {
            let mut selected = genomes.choose_multiple(&mut rand::rng(), 2);
            let first = selected.next().unwrap();
            let second = selected.next().unwrap();
            let mut genome = genome::Genome::crossover(first, second);
            genome.mutate(settings.mutation_chance);

            commands.spawn(SpecimenBundle::new(
                settings.world_size,
                Genome(genome),
                &settings.brain_inputs,
                &settings.brain_outputs,
                settings.internal_neurons,
            ));
        }
    }
}

fn time_system(
    mut turn: ResMut<Turn>,
    mut generation: ResMut<Generation>,
    mut generation_start: ResMut<GenerationStartTime>,
    settings: Res<Settings>,
    mut query: Query<(&mut Age,), With<Alive>>,
) {
    if turn.0 == settings.turns_per_generation {
        generation.0 += 1;
        turn.0 = 0;
        println!("Time: {}", generation_start.0.elapsed().as_secs_f64());
        generation_start.0 = Instant::now();
    } else {
        turn.0 += 1;
        for (mut age,) in query.iter_mut() {
            age.0 += 1;
        }
    }
}

fn thinking_system(mut query: Query<(&mut Brain, &BrainInputs, &mut BrainOutputs), With<Alive>>) {
    query
        .iter_mut()
        .for_each(|(mut brain, brain_inputs, mut brain_outputs)| {
            // TODO: make parallel
            let outputs = brain.0.think(brain_inputs.read());
            *brain_outputs = BrainOutputs::from_hashmap(outputs);
        });
}

fn doing_system(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &BrainOutputs,
            &mut Direction,
            &mut SpeedMultiplier,
            &mut Memory,
        ),
        With<Alive>,
    >,
) {
    use neural_network::Output;
    for (entity, brain_outputs, mut direction, mut speed, mut memory) in &mut query.iter_mut() {
        if brain_outputs.activated(Output::Move) {
            commands.entity(entity).insert(WantsToMove);
        }
        if brain_outputs.activated(Output::Turn) {
            let desired_x = brain_outputs.get(Output::DesiredDirectionX);
            let desired_y = brain_outputs.get(Output::DesiredDirectionY);
            let desired_direction = Vector2::new(desired_x.value(), desired_y.value()).normalize();
            *direction = Direction(Rotation2::rotation_between(
                &Vector2::new(1.0, 0.0),
                &desired_direction,
            ));
        }
        if brain_outputs.activated(Output::ChangeSpeed) {
            let output = brain_outputs.get(Output::DesiredSpeed);
            speed.set_from_neuron_value(&output);
        }
        for i in 0..MEMORY_SIZE {
            if brain_outputs.activated(Output::Remember(i)) {
                let output = brain_outputs.get(Output::DesiredMemory(i));
                memory.0[i] = output.value();
            }
        }
    }
}

fn movement_system(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &mut Position,
            &mut PreviousPosition,
            &SpeedMultiplier,
            &Direction,
        ),
        (With<WantsToMove>, With<Alive>),
    >,
    settings: Res<Settings>,
) {
    for (entity, mut position, mut previous_position, speed, direction) in query.iter_mut() {
        previous_position.0 = position.0;
        position.0 += direction.0 * parry2d::na::Vector2::new(speed.0 * settings.base_speed, 0.0);
        position.0.x = position
            .0
            .x
            .clamp(-settings.world_half_size, settings.world_half_size);
        position.0.y = position
            .0
            .y
            .clamp(-settings.world_half_size, settings.world_half_size);
        commands.entity(entity).remove::<WantsToMove>();
    }
}

fn text_update_system(
    turn: Res<Turn>,
    generation: Res<Generation>,
    mut query: Query<&mut Text, With<TurnText>>,
) {
    for mut text in query.iter_mut() {
        text.0 = format!("Generation: {}\nTurn: {}", generation.0, turn.0);
    }
}

const DISPLAY_SCALE: f32 = 5.0;

fn display_system(_turn: Res<Turn>, mut query: Query<(&Position, &mut Transform)>) {
    for (position, mut transform) in query.iter_mut() {
        // TODO scale
        transform.translation = Vec3::new(
            position.0.x * DISPLAY_SCALE,
            position.0.y * DISPLAY_SCALE,
            0.0,
        );
    }
}

fn setup_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    use bevy::prelude::*;
    commands.spawn(Camera2d);
    commands.insert_resource(Settings::default());
    commands.insert_resource(Turn(0));
    commands.insert_resource(Generation(0));
    commands.insert_resource(GenerationStartTime(Instant::now()));
    commands.spawn((
        Text::new(""),
        TextFont {
            font: asset_server.load("fonts/Roboto-Regular.ttf"),
            font_size: 100.0,
            ..default()
        },
        TextColor(Color::BLACK),
        Node {
            align_self: AlignSelf::FlexEnd,
            position_type: PositionType::Absolute,
            bottom: Val::Px(5.0),
            right: Val::Px(15.0),
            ..Default::default()
        },
        TurnText,
    ));
}

fn is_first_turn(generation: Res<Generation>, turn: Res<Turn>) -> bool {
    generation.0 == 0 && turn.0 == 0
}

fn first_generation_system(mut commands: Commands, settings: Res<Settings>) {
    info!("First generation");
    // Initialize the first generation with random genomes
    for _ in 0..settings.population {
        let genome = genome::Genome::random(settings.genome_length);
        commands.spawn(SpecimenBundle::new(
            settings.world_size,
            Genome(genome),
            &settings.brain_inputs,
            &settings.brain_outputs,
            settings.internal_neurons,
        ));
    }
}

fn damage_system(mut query: Query<(&Position, &mut Health), With<Alive>>, settings: Res<Settings>) {
    for (position, mut health) in query.iter_mut() {
        let damage = if position.0.x <= 0.0 || position.0.y <= 0.0 {
            let distance_from_goal =
                ((-position.0.x).max(0.0).powi(2) + (-position.0.y).max(0.0).powi(2)).sqrt();
            // Convert distance to damage - further means more damage
            4.0 * distance_from_goal / settings.world_half_size
        } else {
            0.0
        };
        health.0 -= damage;
    }
}

fn death_system(mut commands: Commands, query: Query<(Entity, &Health), With<Alive>>) {
    for (entity, health) in query.iter() {
        if health.0 <= 0.0 {
            commands.entity(entity).remove::<Alive>();
        }
    }
}

fn transparency_system(mut query: Query<(&Health, &mut Fill, &mut Stroke, Option<&Alive>)>) {
    for (health, mut fill, mut stroke, alive) in query.iter_mut() {
        match alive {
            Some(_) => {
                let health_value = health.0.clamp(0.0, 100.0);
                let alpha = health_value / 100.0;
                let previous_color = fill.color.to_srgba();
                *fill = Fill::color(Color::srgba(
                    previous_color.red,
                    previous_color.green,
                    previous_color.blue,
                    alpha,
                ));
                *stroke = Stroke::new(Color::BLACK, 1.0);
            }
            None => {
                *fill = Fill::color(Color::NONE);
                *stroke = Stroke::new(Color::BLACK, 2.0);
            }
        }
    }
}
