"""Configuration data structures for the tightrope game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    """Camera placement settings relative to the player."""

    height_offset: float = 2.5
    distance: float = 7.0
    lateral_follow_strength: float = 0.6  # 0..1
    vertical_follow_strength: float = 0.5
    fov: float = 900.0  # pseudo focal length used for perspective projection


@dataclass(frozen=True)
class PhysicsConfig:
    """Tunable physics parameters for the balancing model."""

    gravity: float = 9.81  # m/s^2 equivalent
    lean_inertia: float = 2.4  # rotational inertia around rope axis
    gravity_torque: float = 5.0  # destabilising torque magnitude
    input_torque: float = 12.0  # torque per unit of input
    damping: float = 6.5  # rotational damping coefficient
    max_angle_radians: float = 1.15  # fall threshold (~65.9 degrees)
    max_angular_velocity: float = 4.0  # rad/s clamp to reduce twitchiness
    jump_impulse: float = 5.2  # initial vertical velocity m/s
    jump_air_control: float = 0.35  # reduction factor for torque while airborne
    rope_radius: float = 0.07  # for visual offset
    forward_speed: float = 1.8  # m/s along the rope
    player_height: float = 1.75


@dataclass(frozen=True)
class WindConfig:
    """Controls frequency and strength of random wind gusts."""

    min_interval: float = 2.0
    max_interval: float = 6.0
    min_duration: float = 0.8
    max_duration: float = 2.5
    max_torque: float = 12.0


@dataclass(frozen=True)
class RenderingConfig:
    """Visual parameters for the pseudo-3D renderer."""

    horizon_color: tuple[int, int, int] = (180, 210, 240)
    rope_color: tuple[int, int, int] = (80, 60, 50)
    pole_color: tuple[int, int, int] = (230, 220, 200)
    player_color: tuple[int, int, int] = (200, 50, 50)
    wind_color: tuple[int, int, int] = (80, 120, 255)
    cloud_color: tuple[int, int, int] = (240, 245, 255)
    tree_color: tuple[int, int, int] = (40, 120, 70)
    building_color: tuple[int, int, int] = (160, 170, 190)
    background_color: tuple[int, int, int] = (210, 225, 255)
    ui_color: tuple[int, int, int] = (20, 30, 60)
    chart_bg_color: tuple[int, int, int] = (235, 240, 250)
    chart_line_color: tuple[int, int, int] = (30, 90, 180)
    rope_length: float = 40.0
    rope_segments: int = 18
    rope_width: float = 0.12
    ground_plane: float = -0.5


@dataclass(frozen=True)
class GameConfig:
    """High-level configuration of the game."""

    window_size: tuple[int, int] = (1280, 720)
    target_fps: int = 60
    camera: CameraConfig = field(default_factory=CameraConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    render: RenderingConfig = field(default_factory=RenderingConfig)
