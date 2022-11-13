mod genome;
mod neural_network;
mod settings;
mod specimen;

use crate::settings::{Settings, MEMORY_SIZE};
use crate::specimen::{
    Age, Alive, Birthplace, Brain, BrainInputs, BrainOutputs, Direction, Genome, Memory,
    NeuronValue, NeuronValueConvertible, Oscillator, Position, PreviousPosition, SpecimenBundle,
    SpeedMultiplier,
};
use bevy::app::{App, CoreStage};
use bevy::ecs::prelude::*;
use bevy::prelude::{
    default, AssetServer, Msaa, PluginGroup, Text, Transform, Vec3, WindowDescriptor, WindowPlugin,
};
use bevy::window::PresentMode;
use bevy::DefaultPlugins;
use bevy_prototype_lyon::prelude::*;
use parry2d::na::distance;
use rand::random;
use std::time::Instant;

#[derive(Clone, Hash, Debug, PartialEq, Eq, StageLabel)]
struct GenerationChangeStage;

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                title: "Turbo Evolution Giga Simulator".to_string(),
                width: 550.,
                height: 550.,
                present_mode: PresentMode::Immediate,
                ..default()
            },
            ..default()
        }))
        .add_plugin(ShapePlugin)
        .add_startup_system(setup_system)
        .add_system(display_system)
        .add_system(movement_system)
        .add_system(brain_input_collection_system)
        .add_system(thinking_system)
        .add_system(doing_system)
        .add_system(time_system)
        .add_system(text_update_system)
        .add_stage_before(
            CoreStage::Update,
            GenerationChangeStage,
            SystemStage::parallel(),
        )
        .add_system_to_stage(GenerationChangeStage, new_generation_system)
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
    mut query: Query<(
        &mut BrainInputs,
        &Position,
        &SpeedMultiplier,
        &Direction,
        &Birthplace,
        &Oscillator,
        &PreviousPosition,
        &Memory,
        With<Alive>,
    )>,
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
        oscillator,
        previous_position,
        memory,
        _,
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
        brain_inputs.add(Input::Speed, speed.as_neuron_value());
        brain_inputs.add(Input::Direction, direction.as_neuron_value());
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
    turn: Res<Turn>,
    generation: Res<Generation>,
    to_remove: Query<(Entity, With<Genome>)>,
    alive: Query<(&Genome, &Position, With<Alive>)>,
) {
    if turn.0 == 0 {
        if generation.0 == 0 {
            // Initialize the first generation
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
        } else if alive.iter().count() < settings.population {
            dbg!(alive.iter().count());
            todo!()
        } else {
            // TODO
            let genomes = alive
                .iter()
                .filter(|(_, position, _)| {
                    position.0.x < settings.world_half_size / 2.0
                        && position.0.x > -settings.world_half_size / 2.0
                        && position.0.y < settings.world_half_size / 2.0
                        && position.0.y > -settings.world_half_size / 2.0
                })
                .map(|(genome, _, _)| genome.0.clone())
                .collect::<Vec<_>>();
            println!("Generation {}: {}", generation.0, genomes.len());
            for (entity, _) in to_remove.iter() {
                commands.entity(entity).despawn();
            }

            use rand::seq::SliceRandom;

            for _ in 0..settings.population {
                let mut selected = genomes.choose_multiple(&mut rand::thread_rng(), 2);
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
}

fn time_system(
    mut turn: ResMut<Turn>,
    mut generation: ResMut<Generation>,
    mut generation_start: ResMut<GenerationStartTime>,
    settings: Res<Settings>,
    mut query: Query<(&mut Age,)>,
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

fn thinking_system(mut query: Query<(&mut Brain, &BrainInputs, &mut BrainOutputs)>) {
    query.par_for_each_mut(1, |(mut brain, brain_inputs, mut brain_outputs)| {
        let outputs = brain.0.think(brain_inputs.read());
        *brain_outputs = BrainOutputs::from_hashmap(outputs);
    });
}

fn doing_system(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &BrainOutputs,
        &mut Direction,
        &mut SpeedMultiplier,
        &mut Memory,
    )>,
) {
    use neural_network::Output;
    for (entity, brain_outputs, mut direction, mut speed, mut memory) in &mut query.iter_mut() {
        if brain_outputs.activated(Output::Move) {
            commands.entity(entity).insert(WantsToMove);
        }
        if brain_outputs.activated(Output::Turn) {
            let output = brain_outputs.get(Output::DesiredDirection);
            *direction = Direction::from_neuron_value(&output);
        }
        if brain_outputs.activated(Output::ChangeSpeed) {
            let output = brain_outputs.get(Output::DesiredSpeed);
            *speed = SpeedMultiplier::from_neuron_value(&output);
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
    mut query: Query<(
        Entity,
        &mut Position,
        &mut PreviousPosition,
        &SpeedMultiplier,
        &Direction,
        With<WantsToMove>,
    )>,
    settings: Res<Settings>,
) {
    for (entity, mut position, mut previous_position, speed, direction, _) in query.iter_mut() {
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
        text.sections[0].value = format!("Generation: {}\nTurn: {}", generation.0, turn.0);
    }
}

const DISPLAY_SCALE: f32 = 5.0;

fn display_system(turn: Res<Turn>, mut query: Query<(&Position, &mut Transform)>) {
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
    commands.spawn(Camera2dBundle::new_with_far(100.0));
    commands.insert_resource(Settings::default());
    commands.insert_resource(Turn(0));
    commands.insert_resource(Generation(0));
    commands.insert_resource(GenerationStartTime(Instant::now()));
    commands.spawn((
        TextBundle {
            style: Style {
                align_self: AlignSelf::FlexEnd,
                position_type: PositionType::Absolute,
                position: UiRect {
                    bottom: Val::Px(5.0),
                    right: Val::Px(15.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            text: Text::from_section(
                "",
                TextStyle {
                    font: asset_server.load("fonts/Roboto-Regular.ttf"),
                    font_size: 100.0,
                    color: Color::BLACK,
                },
            ),
            ..Default::default()
        },
        TurnText,
    ));
}
