use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;
use bevy::window::PrimaryWindow;
use bevy_vector_shapes::prelude::*;
use std::collections::HashMap;

use crate::specimen::{Brain, BrainInputs, BrainOutputs, OriginalColor};
use crate::neural_network::{Input, Output};
use crate::settings::Settings;

// Component to mark the selected specimen
#[derive(Component)]
pub struct SelectedSpecimen;

// Resource to track which entity is selected
#[derive(Resource)]
pub struct SelectedSpecimenResource {
    pub entity: Option<Entity>,
}

impl Default for SelectedSpecimenResource {
    fn default() -> Self {
        Self { entity: None }
    }
}

// Resource to store brain visualization data
#[derive(Resource)]
pub struct BrainVisData {
    pub input_positions: HashMap<Input, Vec2>,
    pub neuron_positions: Vec<Vec2>,
    pub output_positions: HashMap<Output, Vec2>,
    pub input_values: HashMap<Input, f32>,
    pub neuron_values: Vec<f32>,
    pub output_values: HashMap<Output, f32>,
    pub connections: Vec<(Vec2, Vec2, f32)>, // (from, to, weight)
}

impl Default for BrainVisData {
    fn default() -> Self {
        Self {
            input_positions: HashMap::new(),
            neuron_positions: Vec::new(),
            output_positions: HashMap::new(),
            input_values: HashMap::new(),
            neuron_values: Vec::new(),
            output_values: HashMap::new(),
            connections: Vec::new(),
        }
    }
}

// Toggle brain visualization with the 'B' key
pub fn toggle_brain_vis_system(keyboard_input: Res<ButtonInput<KeyCode>>, mut settings: ResMut<Settings>) {
    if keyboard_input.just_pressed(KeyCode::KeyB) {
        settings.show_brain_visualization = !settings.show_brain_visualization;
        println!(
            "Brain visualization {}",
            if settings.show_brain_visualization {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

// System to handle mouse clicks and select specimens
pub fn select_specimen_system(
    mouse_input: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut selected: ResMut<SelectedSpecimenResource>,
    specimens: Query<
        (Entity, &Transform, &crate::specimen::Size, &MeshMaterial2d<ColorMaterial>, &OriginalColor),
        With<crate::specimen::Alive>
    >,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if !mouse_input.just_pressed(MouseButton::Left) {
        return;
    }

    let window = if let Ok(win) = windows.get_single() { win } else { return };
    let (camera, camera_transform) = if let Ok(cam) = camera_q.get_single() { cam } else { return };

    let Some(cursor_position) = window.cursor_position() else { return };

    let world_position = camera.viewport_to_world_2d(camera_transform, cursor_position)
        .unwrap_or_default();

    // Clear previous selection
    if let Some(entity) = selected.entity {
        if let Ok((_, _, _, material_handle, original_color)) = specimens.get(entity) {
            if let Some(material) = materials.get_mut(&material_handle.0) {
                material.color = original_color.0; // Revert to original color
            }
        }
    }

    // Check if a specimen is clicked
    selected.entity = None;
    for (entity, transform, size, material_handle, _original_color) in specimens.iter() {
        let pos = transform.translation.truncate();
        if world_position.distance(pos) < size.0 {
            selected.entity = Some(entity);
            if let Some(material) = materials.get_mut(&material_handle.0) {
                material.color = Color::srgb(1.0, 1.0, 0.0); // Yellow highlight
            }
            break;
        }
    }

    println!("Selected specimen: {:?}", selected.entity);
}

// System to update brain visualization data from the selected specimen
pub fn update_brain_vis_data(
    selected: Res<SelectedSpecimenResource>,
    specimens: Query<(&Brain, &BrainInputs, &BrainOutputs)>,
    mut brain_vis_data: ResMut<BrainVisData>,
    settings: Res<Settings>,
) {
    if !settings.show_brain_visualization {
        // Clear data if visualization is off
        brain_vis_data.input_positions.clear();
        brain_vis_data.neuron_positions.clear();
        brain_vis_data.output_positions.clear();
        brain_vis_data.connections.clear();
        brain_vis_data.input_values.clear();
        brain_vis_data.neuron_values.clear();
        brain_vis_data.output_values.clear();
        return;
    }

    if let Some(entity) = selected.entity {
        if let Ok((brain, inputs, outputs)) = specimens.get(entity) {
            // Clear previous data before populating
            brain_vis_data.input_positions.clear();
            brain_vis_data.neuron_positions.clear();
            brain_vis_data.output_positions.clear();
            brain_vis_data.connections.clear();
            brain_vis_data.input_values.clear();
            brain_vis_data.neuron_values.clear();
            brain_vis_data.output_values.clear();

            // Layout positions for visualization
            let window_width = settings.brain_vis_window_width;
            let window_height = settings.brain_vis_window_height;
            
            // Position inputs on the left
            let input_width_offset = 50.0; // Offset from the left edge
            let input_count = settings.brain_inputs.len();
            let input_spacing = window_height / (input_count as f32 + 1.0);
            
            for (i, input_type) in settings.brain_inputs.iter().enumerate() {
                let pos = Vec2::new(
                    -window_width / 2.0 + input_width_offset,
                    window_height / 2.0 - input_spacing * (i as f32 + 1.0) // Y inverted for top-down
                );
                brain_vis_data.input_positions.insert(*input_type, pos);
                
                if let Some(value) = inputs.read().get(input_type) {
                    brain_vis_data.input_values.insert(*input_type, *value);
                }
            }
            
            // Position neurons in the middle
            let neuron_count = settings.internal_neurons;
            let neuron_spacing = window_height / (neuron_count as f32 + 1.0);
            
            for i in 0..neuron_count {
                let pos = Vec2::new(
                    0.0,
                    window_height / 2.0 - neuron_spacing * (i as f32 + 1.0) // Y inverted
                );
                brain_vis_data.neuron_positions.push(pos);
                
                let neuron_values_from_brain = brain.0.get_neuron_values();
                if i < neuron_values_from_brain.len() {
                    brain_vis_data.neuron_values.push(neuron_values_from_brain[i]);
                } else {
                    brain_vis_data.neuron_values.push(0.0); // Default if out of bounds
                }
            }
            
            // Position outputs on the right
            let output_width_offset = 50.0; // Offset from the right edge
            let output_count = settings.brain_outputs.len();
            let output_spacing = window_height / (output_count as f32 + 1.0);
            
            for (i, output_type) in settings.brain_outputs.iter().enumerate() {
                let pos = Vec2::new(
                    window_width / 2.0 - output_width_offset,
                    window_height / 2.0 - output_spacing * (i as f32 + 1.0) // Y inverted
                );
                brain_vis_data.output_positions.insert(*output_type, pos);
                brain_vis_data.output_values.insert(*output_type, outputs.get(*output_type).value());
            }
            
            // Extract connections
            let mut connections = Vec::new();
            for &(input_idx, neuron_idx, weight) in brain.0.get_input_to_internal() {
                if input_idx < settings.brain_inputs.len() && neuron_idx < settings.internal_neurons {
                    let input_type = settings.brain_inputs[input_idx];
                    if let Some(from_pos) = brain_vis_data.input_positions.get(&input_type) {
                        let to_pos = brain_vis_data.neuron_positions[neuron_idx];
                        connections.push((*from_pos, to_pos, weight));
                    }
                }
            }
            
            for &(neuron_idx, output_idx, weight) in brain.0.get_internal_to_output() {
                if neuron_idx < settings.internal_neurons && output_idx < settings.brain_outputs.len() {
                    let output_type = settings.brain_outputs[output_idx];
                    if let Some(to_pos) = brain_vis_data.output_positions.get(&output_type) {
                        let from_pos = brain_vis_data.neuron_positions[neuron_idx];
                        connections.push((from_pos, *to_pos, weight));
                    }
                }
            }
            
            for &(input_idx, output_idx, weight) in brain.0.get_input_to_output() {
                if input_idx < settings.brain_inputs.len() && output_idx < settings.brain_outputs.len() {
                    let input_type = settings.brain_inputs[input_idx];
                    let output_type = settings.brain_outputs[output_idx];
                    if let (Some(from_pos), Some(to_pos)) = (
                        brain_vis_data.input_positions.get(&input_type),
                        brain_vis_data.output_positions.get(&output_type)
                    ) {
                        connections.push((*from_pos, *to_pos, weight));
                    }
                }
            }
            brain_vis_data.connections = connections;
        } else {
            // Selected entity does not have brain components, clear data
            brain_vis_data.input_positions.clear();
            brain_vis_data.neuron_positions.clear();
            brain_vis_data.output_positions.clear();
            brain_vis_data.connections.clear();
            brain_vis_data.input_values.clear();
            brain_vis_data.neuron_values.clear();
            brain_vis_data.output_values.clear();
        }
    } else {
        // No specimen selected, clear data
        brain_vis_data.input_positions.clear();
        brain_vis_data.neuron_positions.clear();
        brain_vis_data.output_positions.clear();
        brain_vis_data.connections.clear();
        brain_vis_data.input_values.clear();
        brain_vis_data.neuron_values.clear();
        brain_vis_data.output_values.clear();
    }
}

// System to render the brain visualization using ShapePainter
pub fn render_brain_visualization_system(
    mut painter: ShapePainter, 
    brain_vis_data: Res<BrainVisData>,
    settings: Res<Settings>,
    asset_server: Res<AssetServer>,
    mut commands: Commands, 
    query_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    if !settings.show_brain_visualization || brain_vis_data.connections.is_empty() && brain_vis_data.input_positions.is_empty() {
        return;
    }
    
    let font = asset_server.load("fonts/Roboto-Regular.ttf");
    let node_radius = 10.0;
    let text_color = Color::BLACK;
    let value_text_color = Color::srgb(0.2, 0.2, 0.2);

    let Ok((_camera, _camera_transform)) = query_camera.get_single() else { return };

    // Use a fixed world position for brain visualization
    // The visualization positions are already calculated relative to (0,0)
    let brain_vis_world_center = Vec2::ZERO;

    painter.set_translation(brain_vis_world_center.extend(10.0));

    // Draw connections
    for (from, to, weight) in &brain_vis_data.connections {
        let weight_clamped = weight.clamp(-1.0, 1.0);
        let line_color = if *weight < 0.0 {
            Color::srgba(1.0, 0.0, 0.0, weight_clamped.abs() * 0.8)
        } else {
            Color::srgba(0.0, 0.8, 0.0, weight_clamped.abs() * 0.8)
        };
        let thickness = 1.0 + 3.0 * weight_clamped.abs();
        
        painter.thickness = thickness;
        painter.color = line_color;
        painter.line(from.extend(0.0), to.extend(0.0));
    }

    // Draw input nodes
    for (input_type, pos) in &brain_vis_data.input_positions {
        let value = brain_vis_data.input_values.get(input_type).unwrap_or(&0.0);
        let intensity = value.abs().min(1.0); 
        let node_color = Color::srgb(0.5 + 0.5 * intensity, 0.5, 0.5 - 0.5 * intensity); 

        painter.color = node_color;
        painter.circle(node_radius);
        
        // Spawn text for label
        commands.spawn((
            Text2d::new(format!("{:?}", input_type)),
            TextFont {
                font: font.clone(),
                font_size: 12.0,
                ..default()
            },
            TextColor(text_color),
            Transform::from_translation((brain_vis_world_center + *pos + Vec2::new(node_radius + 15.0, 0.0)).extend(11.0)),
            BrainVisTextEntity,
        ));
        // Spawn text for value
        commands.spawn((
            Text2d::new(format!("{:.2}", value)),
            TextFont {
                font: font.clone(),
                font_size: 10.0,
                ..default()
            },
            TextColor(value_text_color),
            Transform::from_translation((brain_vis_world_center + *pos - Vec2::new(0.0, node_radius + 10.0)).extend(11.0)),
            BrainVisTextEntity,
        ));
    }

    // Draw neuron nodes
    for (i, pos) in brain_vis_data.neuron_positions.iter().enumerate() {
        let value = brain_vis_data.neuron_values.get(i).unwrap_or(&0.0);
        let intensity = value.abs().min(1.0);
        let node_color = Color::srgb(0.5, 0.5 + 0.5 * intensity, 0.5 - 0.5 * intensity); 

        painter.color = node_color;
        painter.circle(node_radius);

        // Spawn text for value
        commands.spawn((
            Text2d::new(format!("{:.2}", value)),
            TextFont {
                font: font.clone(),
                font_size: 10.0,
                ..default()
            },
            TextColor(value_text_color),
            Transform::from_translation((brain_vis_world_center + *pos - Vec2::new(0.0, node_radius + 10.0)).extend(11.0)),
            BrainVisTextEntity,
        ));
    }

    // Draw output nodes
    for (output_type, pos) in &brain_vis_data.output_positions {
        let value = brain_vis_data.output_values.get(output_type).unwrap_or(&0.0);
        let intensity = value.abs().min(1.0);
        let node_color = Color::srgb(0.5 - 0.5 * intensity, 0.5, 0.5 + 0.5 * intensity);

        painter.color = node_color;
        painter.circle(node_radius);

        // Spawn text for label
        commands.spawn((
            Text2d::new(format!("{:?}", output_type)),
            TextFont {
                font: font.clone(),
                font_size: 12.0,
                ..default()
            },
            TextColor(text_color),
            Transform::from_translation((brain_vis_world_center + *pos + Vec2::new(node_radius + 15.0, 0.0)).extend(11.0)),
            BrainVisTextEntity,
        ));
        // Spawn text for value
        commands.spawn((
            Text2d::new(format!("{:.2}", value)),
            TextFont {
                font: font.clone(),
                font_size: 10.0,
                ..default()
            },
            TextColor(value_text_color),
            Transform::from_translation((brain_vis_world_center + *pos - Vec2::new(0.0, node_radius + 10.0)).extend(11.0)),
            BrainVisTextEntity,
        ));
    }
    
    painter.thickness = 1.0;
    painter.color = Color::WHITE;
}

// Marker component for text entities
#[derive(Component)]
pub struct BrainVisTextEntity; // Marker component for text entities

// System to clean up brain visualization text entities
pub fn cleanup_brain_vis_text_system(
    mut commands: Commands,
    query_text: Query<Entity, With<BrainVisTextEntity>>,
    settings: Res<Settings>,
    selected: Res<SelectedSpecimenResource>,
    // Detect when selection changes or visualization is turned off
    mut last_selected_entity: Local<Option<Entity>>,
    mut last_vis_status: Local<bool>,
) {
    let current_vis_status = settings.show_brain_visualization;
    let current_selected_entity = selected.entity;

    // Despawn text if visualization is turned off OR if the selected entity changes while vis is on
    if !current_vis_status || 
       (current_vis_status && current_selected_entity != *last_selected_entity) {
        if *last_vis_status || (current_selected_entity != *last_selected_entity) {
            // Only despawn if vis was on before, or if selection changed
            for entity in query_text.iter() {
                commands.entity(entity).despawn();
            }
        }
    }
    *last_selected_entity = current_selected_entity;
    *last_vis_status = current_vis_status;
}

// TODO: The `select_specimen_system` needs to be updated to correctly revert the color
// of the previously selected specimen. This might involve storing the original material
// or using a marker component for the highlight and a separate system to manage it.
// The current implementation just sets it to white, which might not be correct.

// TODO: Ensure `Shape2dPlugin::default()` is added to the Bevy App in main.rs.
// e.g. app.add_plugins(Shape2dPlugin::default());

// TODO: The painter's translation in `render_brain_visualization_system` might need adjustment
// depending on how and where the brain visualization is intended to be displayed (e.g., corner of the main window).
// The current `painter.set_translation` is a placeholder.
// If the brain vis is meant to be in a fixed position on the main screen, the coordinates
// in `BrainVisData` should be screen-space, or the painter's transform needs to be set
// to map the BrainVisData's world-space coordinates to the correct screen-space location.
// The current `update_brain_vis_data` calculates positions as if (0,0) is the center of a dedicated area.