use crate::settings::SETTINGS;
use crate::specimen::Specimen;
use cgmath::Point2;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SurfaceType {
    Air,
    Earth,
}

pub struct Board {
    board: Box<dyn Fn(Point2<f32>) -> SurfaceType>,
}

impl Board {
    pub fn new(board: Box<dyn Fn(Point2<f32>) -> SurfaceType>) -> Board {
        Board { board }
    }

    pub fn get_surface_type(&self, point: Point2<f32>) -> SurfaceType {
        if point.x < -SETTINGS.world_half_size
            || point.x > SETTINGS.world_half_size
            || point.y < -SETTINGS.world_half_size
            || point.y > SETTINGS.world_half_size
        {
            return SurfaceType::Earth;
        }
        (self.board)(point)
    }
}

pub struct State {
    board: Board,
    population: Vec<Specimen>,
    turn: usize,
}

mod test {
    #[test]
    fn blocked_outside() {
        use super::*;
        let board = Board::new(Box::new(|_| SurfaceType::Air));
        let point_inside = Point2::new(0.0, 0.0);
        let point_outside = Point2::new(SETTINGS.world_half_size + 1.0, 0.0);
        let point_outside2 = Point2::new(-SETTINGS.world_half_size - 1.0, 0.0);
        let point_outside3 = Point2::new(0.0, SETTINGS.world_half_size + 1.0);
        let point_outside4 = Point2::new(0.0, -SETTINGS.world_half_size - 1.0);
        let point_outside5 = Point2::new(SETTINGS.world_size, SETTINGS.world_size);
        assert_eq!(board.get_surface_type(point_inside), SurfaceType::Air);
        assert_eq!(board.get_surface_type(point_outside), SurfaceType::Earth);
        assert_eq!(board.get_surface_type(point_outside2), SurfaceType::Earth);
        assert_eq!(board.get_surface_type(point_outside3), SurfaceType::Earth);
        assert_eq!(board.get_surface_type(point_outside4), SurfaceType::Earth);
        assert_eq!(board.get_surface_type(point_outside5), SurfaceType::Earth);
    }
}
