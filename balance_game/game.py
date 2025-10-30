"""Core gameplay loop and rendering for the tightrope balance game."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import pygame
from pygame.math import Vector2, Vector3

from .config import CameraConfig, GameConfig, PhysicsConfig, RenderingConfig, WindConfig
from .input import InputProvider, InputState, KeyboardInput


@dataclass
class WindGust:
    """Represents a single wind gust applying torque to the player."""

    torque: float
    duration: float
    elapsed: float = 0.0

    def sample(self, dt: float) -> float:
        """Return the instantaneous torque for the current frame."""
        self.elapsed += dt
        if self.elapsed >= self.duration:
            return 0.0
        # Use a smooth bell-shaped curve for the gust intensity.
        phase = self.elapsed / self.duration
        intensity = math.sin(math.pi * phase)
        return self.torque * intensity

    @property
    def active(self) -> bool:
        return self.elapsed < self.duration

    @property
    def intensity(self) -> float:
        """Normalised progress (0..1) for UI feedback."""
        if self.duration <= 0:
            return 0.0
        return min(1.0, self.elapsed / self.duration)


class WindManager:
    """Creates semi-random wind gusts over time."""

    def __init__(self, config: WindConfig) -> None:
        self.config = config
        self._cooldown = self._next_interval()
        self._current: Optional[WindGust] = None

    def _next_interval(self) -> float:
        return random.uniform(self.config.min_interval, self.config.max_interval)

    def update(self, dt: float) -> tuple[float, Optional[WindGust]]:
        gust = self._current
        torque = 0.0

        if gust and gust.active:
            torque = gust.sample(dt)
        elif gust and not gust.active:
            self._current = None
            gust = None

        if not self._current:
            self._cooldown -= dt
            if self._cooldown <= 0:
                duration = random.uniform(self.config.min_duration, self.config.max_duration)
                magnitude = random.uniform(self.config.max_torque * 0.4, self.config.max_torque)
                direction = random.choice((-1.0, 1.0))
                self._current = WindGust(torque=direction * magnitude, duration=duration)
                self._cooldown = self._next_interval()
                torque = self._current.sample(dt)
                gust = self._current

        return torque, gust

    @property
    def active_gust(self) -> Optional[WindGust]:
        return self._current if self._current and self._current.active else None


@dataclass
class BackgroundFeature:
    """Static scenery element that moves past the player."""

    kind: str
    position: Vector3
    width: float
    height: float
    color: tuple[int, int, int]


class SceneryManager:
    """Keeps a queue of background features for a sense of depth."""

    def __init__(self, render_cfg: RenderingConfig) -> None:
        self.cfg = render_cfg
        self.features: list[BackgroundFeature] = []
        self._populate(initial_forward=0.0)

    def _sample_x(self, minimum: float, maximum: float, exclusion: float = 0.0) -> float:
        if exclusion <= 0.0 or minimum >= maximum:
            return random.uniform(minimum, maximum)

        span_allows_left = minimum < -exclusion
        span_allows_right = maximum > exclusion

        if not (span_allows_left and span_allows_right):
            if span_allows_left:
                return random.uniform(minimum, -exclusion)
            if span_allows_right:
                return random.uniform(exclusion, maximum)
            return random.uniform(minimum, maximum)

        if random.random() < 0.5:
            return random.uniform(minimum, -exclusion)
        return random.uniform(exclusion, maximum)

    def _populate(self, initial_forward: float) -> None:
        for _ in range(26):
            z = initial_forward + random.uniform(10.0, 90.0)
            self.features.append(self._random_feature(z))

    def _random_feature(self, z: float) -> BackgroundFeature:
        kind = random.choices(
            population=("cloud", "tree", "tower"),
            weights=(0.4, 0.35, 0.25),
        )[0]

        if kind == "cloud":
            x = random.uniform(-9.0, 9.0)
            y = random.uniform(4.0, 7.0)
            width = random.uniform(3.0, 5.4)
            height = width * random.uniform(0.45, 0.65)
            color = self.cfg.cloud_color
        elif kind == "tree":
            x = self._sample_x(-5.0, 5.0, exclusion=2.0)
            y = self.cfg.ground_plane + 0.1
            height = random.uniform(3.0, 4.5)
            width = height * random.uniform(0.18, 0.28)
            color = self.cfg.tree_color
        else:  # tower
            x = self._sample_x(-14.0, 14.0, exclusion=5.0)
            y = self.cfg.ground_plane
            height = random.uniform(9.0, 15.0)
            width = height * random.uniform(0.35, 0.55)
            color = self.cfg.building_color

        return BackgroundFeature(
            kind=kind,
            position=Vector3(x, y, z),
            width=width,
            height=height,
            color=color,
        )

    def update(self, player_forward: float) -> None:
        for feature in self.features:
            if feature.position.z < player_forward - 15.0:
                new_z = player_forward + random.uniform(45.0, 95.0)
                refreshed = self._random_feature(new_z)
                feature.kind = refreshed.kind
                feature.position = refreshed.position
                feature.width = refreshed.width
                feature.height = refreshed.height
                feature.color = refreshed.color

    def draw(self, surface: pygame.Surface, camera: "Camera") -> None:
        for feature in self.features:
            self._draw_feature(surface, camera, feature)

    def _draw_feature(self, surface: pygame.Surface, camera: "Camera", feature: BackgroundFeature) -> None:
        pos = feature.position
        relative = pos - camera.position
        if relative.z <= 0.1:
            return

        factor = camera.cfg.fov / relative.z
        screen_x = camera.screen_size.x * 0.5 + relative.x * factor
        screen_y = camera.screen_size.y * 0.55 - relative.y * factor

        if feature.kind == "cloud":
            width_px = feature.width * factor
            height_px = feature.height * factor
            if width_px <= 1 or height_px <= 1:
                return
            rect = pygame.Rect(0, 0, int(width_px), int(height_px))
            rect.center = (int(screen_x), int(screen_y))
            pygame.draw.ellipse(surface, feature.color, rect)
        else:
            base = camera.project(feature.position)
            top = camera.project(feature.position + Vector3(0.0, feature.height, 0.0))
            if not base or not top:
                return
            width_px = max(1.0, feature.width * factor)
            height_px = base.y - top.y

            if feature.kind == "tree":
                trunk_width = max(1, int(width_px * 0.3))
                trunk_height = int(height_px * 0.45)
                trunk_rect = pygame.Rect(0, 0, trunk_width, max(1, trunk_height))
                trunk_rect.midbottom = (int(base.x), int(base.y))
                pygame.draw.rect(surface, (120, 80, 45), trunk_rect)

                canopy_height = height_px - trunk_height * 0.4
                canopy_rect = pygame.Rect(
                    0,
                    0,
                    int(width_px * 1.4),
                    max(1, int(canopy_height)),
                )
                canopy_rect.midbottom = (int(base.x), int(base.y - trunk_height * 0.6))
                pygame.draw.ellipse(surface, feature.color, canopy_rect)
            else:
                rect = pygame.Rect(0, 0, int(width_px), int(height_px))
                rect.midbottom = (int(base.x), int(base.y))
                pygame.draw.rect(surface, feature.color, rect)


@dataclass
class WindParticle:
    """Single wind particle for gust visualisation."""

    position: Vector2
    velocity: Vector2
    life: float
    max_life: float
    size: float

    def update(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.life += dt

    @property
    def alive(self) -> bool:
        return self.life < self.max_life


class WindParticleSystem:
    """Spawns and updates particles during wind gusts."""

    def __init__(self, color: tuple[int, int, int], max_torque: float, screen_size: tuple[int, int]) -> None:
        self.base_color = color
        self.max_torque = max(1.0, abs(max_torque))
        self.screen_size = Vector2(screen_size)
        self.particles: list[WindParticle] = []

    def resize(self, screen_size: tuple[int, int]) -> None:
        self.screen_size = Vector2(screen_size)

    def update(self, dt: float, gust: Optional[WindGust], torque: float, paused: bool) -> None:
        survivors: list[WindParticle] = []
        for particle in self.particles:
            particle.update(dt)
            if (
                particle.alive
                and -80 <= particle.position.x <= self.screen_size.x + 80
                and -80 <= particle.position.y <= self.screen_size.y + 80
            ):
                survivors.append(particle)
        self.particles = survivors

        if paused or not gust or torque == 0.0:
            return

        intensity = max(0.0, min(1.0, abs(torque) / self.max_torque))
        emission_rate = 220.0 * intensity + 70.0
        to_spawn = emission_rate * dt
        count = int(to_spawn)
        if random.random() < to_spawn - count:
            count += 1

        direction = 1.0 if torque > 0 else -1.0
        base_speed = 220.0 + 260.0 * intensity
        half_span = self.screen_size.x * 0.5

        for _ in range(count):
            y = random.uniform(self.screen_size.y * 0.1, self.screen_size.y * 0.9)
            if direction > 0:
                x = random.uniform(-half_span, self.screen_size.x * 0.5)
            else:
                x = random.uniform(self.screen_size.x * 0.5, self.screen_size.x + half_span)

            velocity = Vector2(
                base_speed * direction * random.uniform(0.75, 1.2),
                random.uniform(-65.0, 65.0),
            )
            size = random.uniform(2.5, 5.6) * (0.6 + intensity * 0.7)
            max_life = random.uniform(1.2, 2.0)

            particle = WindParticle(
                position=Vector2(x, y),
                velocity=velocity,
                life=0.0,
                max_life=max_life,
                size=size,
            )
            self.particles.append(particle)

    def draw(self, surface: pygame.Surface) -> None:
        for particle in self.particles:
            remaining = 1.0 - particle.life / particle.max_life
            alpha = max(40, min(180, int(200 * remaining)))
            color = (*self.base_color, alpha)
            pygame.draw.circle(
                surface,
                color,
                (int(particle.position.x), int(particle.position.y)),
                max(1, int(particle.size)),
            )

class Player:
    """Handles the tightrope walker's physics state."""

    def __init__(self, physics: PhysicsConfig) -> None:
        self.cfg = physics
        self.half_height = physics.player_height * 0.5
        self.reset()

    def reset(self) -> None:
        self.lean_angle = 0.0
        self.angular_velocity = 0.0
        self.vertical_velocity = 0.0
        self.vertical_offset = 0.0  # displacement of torso centre relative to rope
        self.forward = 0.0
        self.lateral_offset = 0.0
        self.on_rope = True
        self.fallen = False

    def update(self, dt: float, inputs: InputState, wind_torque: float) -> None:
        if self.fallen:
            return

        torque = self.cfg.gravity_torque * math.sin(self.lean_angle)
        control_torque = inputs.lean * self.cfg.input_torque
        if not self.on_rope:
            control_torque *= self.cfg.jump_air_control
        torque += control_torque
        torque += wind_torque
        torque -= self.angular_velocity * self.cfg.damping

        angular_acc = torque / self.cfg.lean_inertia
        self.angular_velocity += angular_acc * dt
        max_speed = self.cfg.max_angular_velocity
        if max_speed > 0:
            self.angular_velocity = max(-max_speed, min(max_speed, self.angular_velocity))
        self.lean_angle += self.angular_velocity * dt

        lateral_scale = self.cfg.player_height * 0.18
        self.lateral_offset = math.sin(self.lean_angle) * lateral_scale

        if inputs.jump and self.on_rope:
            self.on_rope = False
            self.vertical_velocity = self.cfg.jump_impulse

        gravity = self.cfg.gravity
        if self.on_rope:
            self.vertical_velocity = 0.0
            self.vertical_offset = 0.0
        else:
            self.vertical_velocity -= gravity * dt
            self.vertical_offset += self.vertical_velocity * dt
            if self.vertical_offset <= 0.0:
                self.vertical_offset = 0.0
                self.vertical_velocity = 0.0
                self.on_rope = True

        self.forward += self.cfg.forward_speed * dt

        if abs(self.lean_angle) > self.cfg.max_angle_radians and self.on_rope:
            self.fallen = True

    def world_centre(self) -> Vector3:
        base_height = self.cfg.rope_radius + self.half_height
        return Vector3(self.lateral_offset, base_height + self.vertical_offset, self.forward)

    def feet_position(self) -> Vector3:
        centre = self.world_centre()
        return centre + Vector3(-math.sin(self.lean_angle) * self.half_height, -self.half_height, 0.0)

    def head_position(self) -> Vector3:
        centre = self.world_centre()
        return centre + Vector3(math.sin(self.lean_angle) * self.half_height, self.half_height, 0.0)

    def pole_endpoints(self) -> tuple[Vector3, Vector3]:
        centre = self.world_centre()
        span = self.cfg.player_height * 0.8
        axis = Vector3(math.cos(self.lean_angle), 0.0, 0.0)
        left = centre + axis * (span * 0.5)
        right = centre - axis * (span * 0.5)
        return left, right


class Camera:
    """Simple chase camera with perspective projection."""

    def __init__(self, config: CameraConfig, physics: PhysicsConfig, screen_size: tuple[int, int]) -> None:
        self.cfg = config
        self.physics = physics
        self.position = Vector3(0.0, physics.player_height, -config.distance)
        self.screen_size = Vector2(screen_size)

    def update(self, dt: float, player: Player) -> None:
        target_centre = player.world_centre()
        desired_x = player.lateral_offset * self.cfg.lateral_follow_strength
        desired_y = target_centre.y + self.cfg.height_offset
        desired_z = target_centre.z - self.cfg.distance

        lerp = min(1.0, 4.0 * dt)
        self.position.x += (desired_x - self.position.x) * lerp
        self.position.y += (desired_y - self.position.y) * (lerp * self.cfg.vertical_follow_strength)
        self.position.z += (desired_z - self.position.z) * lerp

    def project(self, point: Vector3) -> Optional[Vector2]:
        relative = point - self.position
        if relative.z <= 0.1:
            return None
        factor = self.cfg.fov / relative.z
        screen = Vector2(
            self.screen_size.x * 0.5 + relative.x * factor,
            self.screen_size.y * 0.55 - relative.y * factor,
        )
        return screen


class Renderer:
    """Handles all rendering operations."""

    def __init__(
        self,
        surface: pygame.Surface,
        config: RenderingConfig,
        camera: Camera,
        player: Player,
        physics: PhysicsConfig,
        scenery: SceneryManager,
        wind: WindConfig,
    ) -> None:
        self.surface = surface
        self.cfg = config
        self.camera = camera
        self.player = player
        self.physics = physics
        self.scenery = scenery
        self.wind_cfg = wind
        self.wind_particles = WindParticleSystem(self.cfg.wind_color, wind.max_torque, self.surface.get_size())
        self.font = pygame.font.Font(None, 32)
        self.large_font = pygame.font.Font(None, 64)

    def draw(
        self,
        dt: float,
        gust: Optional[WindGust],
        wind_torque: float,
        fps: float,
        waiting_for_start: bool,
        lean_history: list[float],
    ) -> None:
        self.wind_particles.resize(self.surface.get_size())
        self.surface.fill(self.cfg.background_color)
        self._draw_horizon()
        self.scenery.draw(self.surface, self.camera)
        self._draw_rope()
        self._draw_player()
        self.wind_particles.update(dt, gust, wind_torque, waiting_for_start)
        self.wind_particles.draw(self.surface)
        self._draw_ui(gust, fps, waiting_for_start)
        self._draw_lean_chart(lean_history)

    def _draw_horizon(self) -> None:
        width, height = self.surface.get_size()
        pygame.draw.rect(
            self.surface,
            self.cfg.horizon_color,
            pygame.Rect(0, 0, width, int(height * 0.6)),
        )

    def _draw_rope(self) -> None:
        segments = self.cfg.rope_segments
        seg_length = self.cfg.rope_length / segments
        base_z = self.player.forward - seg_length * 2
        left_points: list[Vector2] = []
        right_points: list[Vector2] = []

        for i in range(segments + 3):
            z = base_z + i * seg_length
            y = self.physics.rope_radius
            left_world = Vector3(-self.cfg.rope_width * 0.5, y, z)
            right_world = Vector3(self.cfg.rope_width * 0.5, y, z)

            left_proj = self.camera.project(left_world)
            right_proj = self.camera.project(right_world)
            if left_proj and right_proj:
                left_points.append(left_proj)
                right_points.append(right_proj)

        if len(left_points) >= 2:
            for i in range(len(left_points) - 1):
                quad = [
                    left_points[i],
                    right_points[i],
                    right_points[i + 1],
                    left_points[i + 1],
                ]
                pygame.draw.polygon(self.surface, self.cfg.rope_color, quad)

    def _draw_player(self) -> None:
        head = self.player.head_position()
        feet = self.player.feet_position()
        head_proj = self.camera.project(head)
        feet_proj = self.camera.project(feet)
        if head_proj and feet_proj:
            pygame.draw.line(self.surface, self.cfg.player_color, head_proj, feet_proj, 10)

        left_pole, right_pole = self.player.pole_endpoints()
        left_proj = self.camera.project(left_pole)
        right_proj = self.camera.project(right_pole)
        if left_proj and right_proj:
            pygame.draw.line(self.surface, self.cfg.pole_color, left_proj, right_proj, 4)

        # Draw a small marker at the feet for better depth perception.
        centre_proj = self.camera.project(self.player.world_centre())
        if centre_proj:
            pygame.draw.circle(self.surface, self.cfg.player_color, centre_proj, 12)

    def _draw_wind(self, gust: Optional[WindGust]) -> None:
        if not gust:
            return
        intensity = gust.intensity
        width, height = self.surface.get_size()
        alpha = int(120 * (1.0 - abs(0.5 - intensity) * 2))
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        color = (*self.cfg.wind_color, alpha)
        overlay.fill(color)
        self.surface.blit(overlay, (0, 0))

    def _draw_ui(self, gust: Optional[WindGust], fps: float, waiting_for_start: bool) -> None:
        lean_deg = math.degrees(self.player.lean_angle)
        texts = [
            f"Lean: {lean_deg:+05.1f}°",
            f"Forward: {self.player.forward:05.1f} m",
            f"FPS: {fps:04.1f}",
        ]
        if gust:
            texts.append("Wind gust!")

        for idx, text in enumerate(texts):
            surf = self.font.render(text, True, self.cfg.ui_color)
            self.surface.blit(surf, (24, 24 + idx * 28))

        if waiting_for_start:
            instructions = [
                "Press any key to begin balancing.",
                "Controls: A/Left = lean left, D/Right = lean right, Space = jump",
            ]
        else:
            instructions = [
                "Controls: A/Left = lean left, D/Right = lean right, Space = jump",
                "Press R to reset at any time.",
            ]
        width, height = self.surface.get_size()
        for idx, line in enumerate(instructions):
            surf = self.font.render(line, True, self.cfg.ui_color)
            rect = surf.get_rect()
            rect.bottomleft = (24, height - 24 - idx * 26)
            self.surface.blit(surf, rect)

    def _draw_lean_chart(self, history: list[float]) -> None:
        if not history:
            return

        chart_width = 240
        chart_height = 140
        margin = 24
        rect = pygame.Rect(
            self.surface.get_width() - chart_width - margin,
            margin,
            chart_width,
            chart_height,
        )

        pygame.draw.rect(self.surface, self.cfg.chart_bg_color, rect, border_radius=10)
        pygame.draw.rect(self.surface, self.cfg.ui_color, rect, width=1, border_radius=10)

        mid_y = rect.centery
        pygame.draw.line(
            self.surface,
            self.cfg.ui_color,
            (rect.left + 8, mid_y),
            (rect.right - 8, mid_y),
            1,
        )

        max_angle = math.degrees(self.physics.max_angle_radians)
        if max_angle <= 0:
            return

        samples = history[-160:]
        if len(samples) < 2:
            return

        available_width = rect.width - 16
        step_x = available_width / (len(samples) - 1)
        scale_y = (rect.height * 0.5 - 8) / max_angle

        points: list[tuple[int, int]] = []
        for idx, angle in enumerate(samples):
            deg = math.degrees(angle)
            x = rect.left + 8 + idx * step_x
            y = mid_y - deg * scale_y
            y = max(rect.top + 8, min(rect.bottom - 8, y))
            points.append((int(x), int(y)))

        pygame.draw.lines(self.surface, self.cfg.chart_line_color, False, points, 2)
        current_deg = math.degrees(samples[-1])
        label = self.font.render(f"{current_deg:+04.1f}°", True, self.cfg.ui_color)
        label_rect = label.get_rect()
        label_rect.bottomright = (rect.right - 12, rect.bottom - 10)
        self.surface.blit(label, label_rect)


class TightropeGame:
    """High-level game orchestration."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        input_provider: Optional[InputProvider] = None,
    ) -> None:
        pygame.init()
        pygame.font.init()

        self.config = config or GameConfig()
        self.screen = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption("G5 Tightrope Balance")

        self.clock = pygame.time.Clock()
        self.physics = self.config.physics
        self.player = Player(self.physics)
        self.camera = Camera(self.config.camera, self.physics, self.config.window_size)
        self.scenery = SceneryManager(self.config.render)
        self.renderer = Renderer(
            self.screen,
            self.config.render,
            self.camera,
            self.player,
            self.physics,
            self.scenery,
            self.config.wind,
        )
        self.wind = WindManager(self.config.wind)
        self.input_provider = input_provider or KeyboardInput()

        self.running = True
        self.game_over = False
        self.waiting_for_start = True
        self.current_wind_torque = 0.0
        self.max_history_samples = max(90, int(self.config.target_fps * 6))
        self.lean_history: list[float] = []
        self._record_lean(0.0)

    def reset(self) -> None:
        self.player.reset()
        self.wind = WindManager(self.config.wind)
        self.camera = Camera(self.config.camera, self.physics, self.config.window_size)
        self.scenery = SceneryManager(self.config.render)
        self.renderer = Renderer(
            self.screen,
            self.config.render,
            self.camera,
            self.player,
            self.physics,
            self.scenery,
            self.config.wind,
        )
        if hasattr(self.input_provider, "reset"):
            self.input_provider.reset()  # type: ignore[attr-defined]
        self.game_over = False
        self.waiting_for_start = True
        self.current_wind_torque = 0.0
        self.lean_history = []
        self._record_lean(0.0)

    def _record_lean(self, angle: float) -> None:
        self.lean_history.append(angle)
        if len(self.lean_history) > self.max_history_samples:
            self.lean_history.pop(0)

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.config.target_fps) / 1000.0
            fps = self.clock.get_fps()
            self._handle_events()

            wind_torque = 0.0
            gust: Optional[WindGust] = None

            if not self.waiting_for_start:
                wind_torque, gust = self.wind.update(dt)
                gust = gust or self.wind.active_gust
            else:
                gust = None

            if self.waiting_for_start:
                self.camera.update(dt, self.player)
            elif not self.game_over:
                inputs = self.input_provider.poll(dt)
                self.player.update(dt, inputs, wind_torque)
                self.camera.update(dt, self.player)
                if self.player.fallen:
                    self.game_over = True
            else:
                self.camera.update(dt, self.player)

            if not self.waiting_for_start:
                self.scenery.update(self.player.forward)

            self.current_wind_torque = wind_torque if not self.waiting_for_start else 0.0
            current_angle = 0.0 if self.waiting_for_start else self.player.lean_angle
            self._record_lean(current_angle)

            self.renderer.draw(
                dt,
                gust if not self.waiting_for_start else None,
                self.current_wind_torque,
                fps,
                self.waiting_for_start,
                self.lean_history,
            )

            if self.waiting_for_start:
                self._draw_start_prompt()
            elif self.game_over:
                self._draw_game_over()

            pygame.display.flip()

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif self.waiting_for_start:
                    self.waiting_for_start = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif self.game_over and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self.reset()

    def _draw_game_over(self) -> None:
        width, height = self.screen.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        text = self.renderer.large_font.render("You fell! Press Space or R to retry.", True, (240, 240, 240))
        text_rect = text.get_rect(center=(width // 2, height // 2))
        self.screen.blit(text, text_rect)

    def _draw_start_prompt(self) -> None:
        width, height = self.screen.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title = self.renderer.large_font.render("Press any key to start balancing", True, (245, 245, 245))
        title_rect = title.get_rect(center=(width // 2, height // 2 - 20))
        self.screen.blit(title, title_rect)

        subtitle = self.renderer.font.render("Lean with A/D, jump with Space, press R to reset.", True, (225, 225, 225))
        subtitle_rect = subtitle.get_rect(center=(width // 2, height // 2 + 28))
        self.screen.blit(subtitle, subtitle_rect)
