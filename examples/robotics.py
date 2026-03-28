#!/usr/bin/env python3
"""
Robotics Examples for Constraint Theory.

This example demonstrates how Constraint Theory can be used in robotics
applications for deterministic motion planning, navigation, and control.

Key benefits for robotics:
- Exact position tracking without drift
- Deterministic path planning
- Reproducible robot behavior
- Precise angle calculations
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class SimulatedManifold:
    """Simulated PythagoreanManifold for demo purposes."""
    
    def __init__(self, density: int):
        self.density = density
        self._states = self._generate_states()
    
    def _generate_states(self) -> List[Tuple[float, float]]:
        """Generate Pythagorean triple states."""
        states = []
        triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61),
            (28, 45, 53), (33, 56, 65), (16, 63, 65), (48, 55, 73),
            (15, 8, 17), (24, 7, 25),  # Swapped for more coverage
        ]
        
        for a, b, c in triples:
            for sx in [1, -1]:
                for sy in [1, -1]:
                    states.append((sx * a / c, sy * b / c))
                    states.append((sx * b / c, sy * a / c))
        
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            states.append(d)
        
        return states
    
    def snap(self, x: float, y: float) -> Tuple[float, float, float]:
        """Snap to nearest Pythagorean state."""
        mag = math.sqrt(x * x + y * y)
        if mag == 0:
            return (0.0, 0.0, 0.0)
        
        nx, ny = x / mag, y / mag
        best_state = (0.0, 0.0)
        best_dist = float('inf')
        
        for sx, sy in self._states:
            dist = (nx - sx) ** 2 + (ny - sy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_state = (sx, sy)
        
        noise = math.sqrt(best_dist)
        return (best_state[0], best_state[1], noise)
    
    def snap_angle(self, angle_degrees: float) -> Tuple[float, float, float]:
        """Snap an angle to nearest Pythagorean direction."""
        angle_rad = math.radians(angle_degrees)
        x, y = math.cos(angle_rad), math.sin(angle_rad)
        sx, sy, noise = self.snap(x, y)
        return sx, sy, noise
    
    def get_angle_from_snap(self, x: float, y: float) -> float:
        """Get angle in degrees from snapped coordinates."""
        return math.degrees(math.atan2(y, x))


@dataclass
class Point2D:
    """2D point."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)


@dataclass
class RobotState:
    """Robot state with position and heading."""
    position: Point2D
    heading: float  # degrees
    velocity: float


# =============================================================================
# Example 1: Differential Drive Navigation
# =============================================================================

class DifferentialDriveRobot:
    """
    Differential drive robot with exact heading control.
    
    Uses Constraint Theory for:
    - Exact heading angles
    - Deterministic turning
    - Drift-free navigation
    """
    
    def __init__(self, manifold: SimulatedManifold, 
                 wheel_base: float = 0.5, 
                 max_speed: float = 1.0):
        self.manifold = manifold
        self.wheel_base = wheel_base
        self.max_speed = max_speed
        
        # Robot state
        self.position = Point2D(0, 0)
        self.heading = 0.0  # degrees
        self.velocity = 0.0
        
        # Path history for visualization
        self.path_history: List[Point2D] = []
    
    def set_heading_exact(self, target_heading: float) -> Tuple[float, float]:
        """
        Set heading to exact Pythagorean direction.
        
        Returns: (actual_heading, snap_noise)
        """
        # Snap target heading to exact direction
        sx, sy, noise = self.manifold.snap_angle(target_heading)
        
        # Calculate actual heading from snapped direction
        self.heading = self.manifold.get_angle_from_snap(sx, sy)
        
        return self.heading, noise
    
    def set_velocity(self, velocity: float) -> None:
        """Set robot velocity."""
        self.velocity = max(-self.max_speed, min(self.max_speed, velocity))
    
    def move_toward(self, target: Point2D) -> Tuple[float, float]:
        """
        Calculate exact heading toward target.
        
        Returns: (heading, distance)
        """
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            # Snap direction to exact
            sx, sy, _ = self.manifold.snap(dx, dy)
            self.heading = self.manifold.get_angle_from_snap(sx, sy)
        
        return self.heading, distance
    
    def update(self, dt: float) -> None:
        """Update robot position based on heading and velocity."""
        # Convert heading to radians
        heading_rad = math.radians(self.heading)
        
        # Calculate displacement
        dx = self.velocity * math.cos(heading_rad) * dt
        dy = self.velocity * math.sin(heading_rad) * dt
        
        # Update position
        self.position = Point2D(self.position.x + dx, self.position.y + dy)
        
        # Record path
        self.path_history.append(Point2D(self.position.x, self.position.y))
    
    def navigate_to(self, target: Point2D, dt: float = 0.1) -> List[Point2D]:
        """
        Navigate to target position with exact waypoints.
        
        Simulates navigation and returns path taken.
        """
        path = [Point2D(self.position.x, self.position.y)]
        
        while self.position.distance_to(target) > 0.01:
            # Face target with exact heading
            self.move_toward(target)
            self.set_velocity(self.max_speed)
            self.update(dt)
            path.append(Point2D(self.position.x, self.position.y))
            
            # Safety limit
            if len(path) > 1000:
                break
        
        return path


def demo_differential_drive():
    """Demonstrate differential drive navigation."""
    print("=" * 60)
    print("Example 1: Differential Drive Navigation")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    robot = DifferentialDriveRobot(manifold, wheel_base=0.5, max_speed=1.0)
    
    print("\nNavigating to waypoints with exact headings:")
    print("-" * 40)
    
    waypoints = [
        Point2D(0, 0),
        Point2D(3, 4),    # Classic 3-4-5 triangle
        Point2D(8, 6),    # Another Pythagorean
        Point2D(5, 12),   # 5-12-13 triple
        Point2D(0, 0),    # Return home
    ]
    
    for i, target in enumerate(waypoints):
        heading, distance = robot.move_toward(target)
        print(f"\n   Waypoint {i+1}: ({target.x:.1f}, {target.y:.1f})")
        print(f"   Distance: {distance:.2f}m")
        print(f"   Exact heading: {heading:.2f}°")
        
        # Navigate
        path = robot.navigate_to(target)
        print(f"   Steps taken: {len(path)}")


# =============================================================================
# Example 2: Arm Kinematics
# =============================================================================

class RobotArm:
    """
    Robot arm with exact joint angles.
    
    Uses Constraint Theory for:
    - Exact joint positions
    - Deterministic inverse kinematics
    - Precise end effector positioning
    """
    
    def __init__(self, manifold: SimulatedManifold, 
                 link_lengths: List[float]):
        self.manifold = manifold
        self.link_lengths = link_lengths
        self.joint_angles: List[float] = [0.0] * len(link_lengths)
        self.joint_positions: List[Point2D] = []
        self._update_positions()
    
    def _update_positions(self) -> None:
        """Calculate joint positions from angles."""
        self.joint_positions = [Point2D(0, 0)]  # Base
        
        x, y = 0.0, 0.0
        cumulative_angle = 0.0
        
        for i, (length, angle) in enumerate(zip(self.link_lengths, self.joint_angles)):
            cumulative_angle += angle
            x += length * math.cos(math.radians(cumulative_angle))
            y += length * math.sin(math.radians(cumulative_angle))
            self.joint_positions.append(Point2D(x, y))
    
    def get_end_effector(self) -> Point2D:
        """Get end effector position."""
        return self.joint_positions[-1]
    
    def set_joint_exact(self, joint_index: int, target_angle: float) -> float:
        """
        Set joint angle to exact Pythagorean direction.
        
        Returns: actual angle after snapping
        """
        # Snap target angle to exact
        sx, sy, _ = self.manifold.snap_angle(target_angle)
        actual_angle = self.manifold.get_angle_from_snap(sx, sy)
        
        self.joint_angles[joint_index] = actual_angle
        self._update_positions()
        
        return actual_angle
    
    def inverse_kinematics_2d(self, target: Point2D) -> Optional[Tuple[float, float]]:
        """
        Solve IK for 2-link arm using exact angles.
        
        Returns: (angle1, angle2) or None if unreachable
        """
        if len(self.link_lengths) != 2:
            raise ValueError("IK only implemented for 2-link arm")
        
        L1, L2 = self.link_lengths
        
        # Distance to target
        d = math.sqrt(target.x**2 + target.y**2)
        
        # Check reachability
        if d > L1 + L2 or d < abs(L1 - L2):
            return None
        
        # Law of cosines for elbow angle
        cos_angle2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle2 = max(-1, min(1, cos_angle2))  # Clamp for numerical stability
        
        angle2 = math.degrees(math.acos(cos_angle2))
        
        # Shoulder angle
        angle1 = math.degrees(math.atan2(target.y, target.x)) - \
                 math.degrees(math.atan2(L2 * math.sin(math.radians(angle2)),
                                        L1 + L2 * math.cos(math.radians(angle2))))
        
        # Snap to exact angles
        sx1, sy1, _ = self.manifold.snap_angle(angle1)
        sx2, sy2, _ = self.manifold.snap_angle(angle2)
        
        exact_angle1 = self.manifold.get_angle_from_snap(sx1, sy1)
        exact_angle2 = self.manifold.get_angle_from_snap(sx2, sy2)
        
        return exact_angle1, exact_angle2
    
    def move_to(self, target: Point2D) -> bool:
        """
        Move end effector to target using exact IK.
        
        Returns: True if successful
        """
        if len(self.link_lengths) == 2:
            result = self.inverse_kinematics_2d(target)
            if result:
                self.joint_angles[0] = result[0]
                self.joint_angles[1] = result[1]
                self._update_positions()
                return True
        return False


def demo_robot_arm():
    """Demonstrate robot arm kinematics."""
    print("\n" + "=" * 60)
    print("Example 2: Robot Arm Kinematics")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    arm = RobotArm(manifold, link_lengths=[1.0, 0.8])
    
    print("\n2-DOF arm inverse kinematics:")
    print("-" * 40)
    
    targets = [
        Point2D(1.0, 0.5),
        Point2D(0.5, 1.0),
        Point2D(1.2, 0.6),  # 3-4-5 ratio
        Point2D(0.8, 0.6),
    ]
    
    for target in targets:
        success = arm.move_to(target)
        end = arm.get_end_effector()
        error = target.distance_to(end)
        
        print(f"\n   Target: ({target.x:.2f}, {target.y:.2f})")
        print(f"   Joint angles: [{arm.joint_angles[0]:.2f}°, {arm.joint_angles[1]:.2f}°]")
        print(f"   End effector: ({end.x:.4f}, {end.y:.4f})")
        print(f"   Position error: {error:.4f}")


# =============================================================================
# Example 3: Path Planning
# =============================================================================

class PathPlanner:
    """
    Path planner with exact waypoint directions.
    
    Uses Constraint Theory for:
    - Exact direction vectors between waypoints
    - Deterministic path following
    - Smooth transitions
    """
    
    def __init__(self, manifold: SimulatedManifold):
        self.manifold = manifold
        self.waypoints: List[Point2D] = []
        self.exact_directions: List[Tuple[float, float]] = []
    
    def add_waypoint(self, point: Point2D) -> None:
        """Add waypoint and calculate exact direction from previous."""
        self.waypoints.append(point)
        
        if len(self.waypoints) > 1:
            prev = self.waypoints[-2]
            dx = point.x - prev.x
            dy = point.y - prev.y
            
            # Snap to exact direction
            sx, sy, _ = self.manifold.snap(dx, dy)
            self.exact_directions.append((sx, sy))
    
    def generate_grid_path(self, start: Point2D, end: Point2D, 
                          grid_size: float = 1.0) -> List[Point2D]:
        """
        Generate path using grid-based planning with exact directions.
        
        Creates waypoints that follow exact Pythagorean directions.
        """
        self.waypoints = [start]
        self.exact_directions = []
        
        current = start
        while current.distance_to(end) > grid_size:
            # Direction toward end
            dx = end.x - current.x
            dy = end.y - current.y
            
            # Snap to exact direction
            sx, sy, noise = self.manifold.snap(dx, dy)
            
            # Move one grid unit in exact direction
            next_point = Point2D(
                current.x + sx * grid_size,
                current.y + sy * grid_size
            )
            
            self.waypoints.append(next_point)
            self.exact_directions.append((sx, sy))
            current = next_point
        
        self.waypoints.append(end)
        return self.waypoints
    
    def generate_smooth_path(self, start: Point2D, end: Point2D,
                            num_points: int = 10) -> List[Point2D]:
        """
        Generate smooth path with exact intermediate directions.
        """
        self.waypoints = [start]
        self.exact_directions = []
        
        for i in range(1, num_points + 1):
            t = i / (num_points + 1)
            
            # Linear interpolation
            interp_x = start.x + t * (end.x - start.x)
            interp_y = start.y + t * (end.y - start.y)
            
            # Calculate exact direction at this point
            dx = end.x - interp_x
            dy = end.y - interp_y
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            self.waypoints.append(Point2D(interp_x, interp_y))
            self.exact_directions.append((sx, sy))
        
        self.waypoints.append(end)
        return self.waypoints


def demo_path_planning():
    """Demonstrate path planning."""
    print("\n" + "=" * 60)
    print("Example 3: Path Planning")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    planner = PathPlanner(manifold)
    
    print("\nGrid-based path planning:")
    print("-" * 40)
    
    start = Point2D(0, 0)
    end = Point2D(5, 5)
    
    path = planner.generate_grid_path(start, end, grid_size=1.0)
    
    print(f"\n   From ({start.x}, {start.y}) to ({end.x}, {end.y})")
    print(f"\n   Waypoints ({len(path)} total):")
    
    for i, (wp, direction) in enumerate(zip(
        path, 
        planner.exact_directions + [(0, 0)]
    )):
        if i < len(path) - 1:
            dx, dy = direction
            angle = math.degrees(math.atan2(dy, dx))
            print(f"   [{i}] ({wp.x:.2f}, {wp.y:.2f}) -> direction: ({dx:.4f}, {dy:.4f}), angle: {angle:.1f}°")
        else:
            print(f"   [{i}] ({wp.x:.2f}, {wp.y:.2f}) [END]")


# =============================================================================
# Example 4: Sensor Processing
# =============================================================================

class SensorProcessor:
    """
    Sensor data processing with exact directions.
    
    Uses Constraint Theory for:
    - Exact bearing calculations
    - Deterministic sensor fusion
    - Noise reduction
    """
    
    def __init__(self, manifold: SimulatedManifold):
        self.manifold = manifold
    
    def process_lidar_reading(self, angle: float, distance: float) -> Tuple[Point2D, float]:
        """
        Process LIDAR reading with exact angle.
        
        Returns: (point, snap_error)
        """
        # Snap angle to exact
        sx, sy, noise = self.manifold.snap_angle(angle)
        exact_angle = self.manifold.get_angle_from_snap(sx, sy)
        
        # Calculate point
        point = Point2D(
            distance * math.cos(math.radians(exact_angle)),
            distance * math.sin(math.radians(exact_angle))
        )
        
        return point, noise
    
    def process_imu_heading(self, heading: float) -> Tuple[float, float]:
        """
        Process IMU heading with exact direction.
        
        Returns: (exact_heading, snap_error)
        """
        sx, sy, noise = self.manifold.snap_angle(heading)
        exact_heading = self.manifold.get_angle_from_snap(sx, sy)
        return exact_heading, noise
    
    def triangulate_position(self, bearing1: float, bearing2: float,
                            beacon1: Point2D, beacon2: Point2D) -> Optional[Point2D]:
        """
        Triangulate position from exact bearings.
        """
        # Snap bearings to exact
        sx1, sy1, _ = self.manifold.snap_angle(bearing1)
        sx2, sy2, _ = self.manifold.snap_angle(bearing2)
        
        angle1 = math.radians(self.manifold.get_angle_from_snap(sx1, sy1))
        angle2 = math.radians(self.manifold.get_angle_from_snap(sx2, sy2))
        
        # Line from beacon 1
        x1, y1 = beacon1.x, beacon1.y
        dx1, dy1 = math.cos(angle1), math.sin(angle1)
        
        # Line from beacon 2
        x2, y2 = beacon2.x, beacon2.y
        dx2, dy2 = math.cos(angle2), math.sin(angle2)
        
        # Solve for intersection
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-10:
            return None  # Parallel lines
        
        t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
        
        return Point2D(x1 + t * dx1, y1 + t * dy1)


def demo_sensor_processing():
    """Demonstrate sensor processing."""
    print("\n" + "=" * 60)
    print("Example 4: Sensor Processing")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    sensor = SensorProcessor(manifold)
    
    print("\nLIDAR readings with exact angles:")
    print("-" * 40)
    
    # Simulated LIDAR scan
    angles = [0, 30, 45, 60, 90, 120, 150, 180]
    distances = [5.0, 4.5, 3.8, 4.2, 6.0, 5.5, 4.0, 3.5]
    
    print(f"\n   {'Angle':<8} {'Distance':<10} {'Point':<20} {'Error':<8}")
    print(f"   {'-'*46}")
    
    for angle, dist in zip(angles, distances):
        point, error = sensor.process_lidar_reading(angle, dist)
        print(f"   {angle:<8.0f} {dist:<10.1f} ({point.x:.2f}, {point.y:.2f}){'':<6} {error:.4f}")
    
    print("\nTriangulation:")
    print("-" * 40)
    
    beacon1 = Point2D(0, 0)
    beacon2 = Point2D(10, 0)
    
    # Bearings from unknown position
    bearing1 = 45.0   # 45 degrees from beacon 1
    bearing2 = 135.0  # 135 degrees from beacon 2
    
    position = sensor.triangulate_position(bearing1, bearing2, beacon1, beacon2)
    
    if position:
        print(f"\n   Beacons at: ({beacon1.x}, {beacon1.y}) and ({beacon2.x}, {beacon2.y})")
        print(f"   Bearings: {bearing1}° and {bearing2}°")
        print(f"   Triangulated position: ({position.x:.4f}, {position.y:.4f})")


# =============================================================================
# Example 5: Formation Control
# =============================================================================

class FormationController:
    """
    Multi-robot formation control with exact relative positions.
    
    Uses Constraint Theory for:
    - Exact formation shapes
    - Deterministic relative positioning
    - Coordinated movement
    """
    
    def __init__(self, manifold: SimulatedManifold, num_robots: int):
        self.manifold = manifold
        self.num_robots = num_robots
        self.robots: List[Point2D] = [Point2D(0, 0) for _ in range(num_robots)]
        self.formation_offsets: List[Point2D] = []
    
    def set_line_formation(self, spacing: float = 1.0) -> None:
        """Set line formation with exact spacing."""
        self.formation_offsets = []
        center = (self.num_robots - 1) / 2
        
        for i in range(self.num_robots):
            offset = i - center
            # Use exact horizontal direction
            self.formation_offsets.append(Point2D(offset * spacing, 0))
    
    def set_wedge_formation(self, spacing: float = 1.0) -> None:
        """Set wedge/V formation with exact angles."""
        self.formation_offsets = []
        
        for i in range(self.num_robots):
            row = i // 2
            side = 1 if i % 2 == 0 else -1
            
            if i == 0:
                # Leader at front
                self.formation_offsets.append(Point2D(0, 0))
            else:
                # Followers at exact 45-degree angles
                sx, sy, _ = self.manifold.snap_angle(45 * side)
                angle = self.manifold.get_angle_from_snap(sx, sy)
                distance = row * spacing
                
                self.formation_offsets.append(Point2D(
                    distance * math.cos(math.radians(angle)),
                    distance * math.sin(math.radians(angle))
                ))
    
    def set_circle_formation(self, radius: float = 2.0) -> None:
        """Set circle formation with exact angular spacing."""
        self.formation_offsets = []
        
        for i in range(self.num_robots):
            angle = (i / self.num_robots) * 360
            sx, sy, _ = self.manifold.snap_angle(angle)
            exact_angle = self.manifold.get_angle_from_snap(sx, sy)
            
            self.formation_offsets.append(Point2D(
                radius * math.cos(math.radians(exact_angle)),
                radius * math.sin(math.radians(exact_angle))
            ))
    
    def get_formation_positions(self, center: Point2D, heading: float) -> List[Point2D]:
        """
        Get absolute positions for formation centered at point with heading.
        """
        positions = []
        heading_rad = math.radians(heading)
        
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        
        for offset in self.formation_offsets:
            # Rotate offset by heading
            rx = offset.x * cos_h - offset.y * sin_h
            ry = offset.x * sin_h + offset.y * cos_h
            
            positions.append(Point2D(center.x + rx, center.y + ry))
        
        return positions


def demo_formation_control():
    """Demonstrate formation control."""
    print("\n" + "=" * 60)
    print("Example 5: Formation Control")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    controller = FormationController(manifold, num_robots=5)
    
    center = Point2D(10, 10)
    heading = 45.0
    
    print("\nLine formation:")
    print("-" * 40)
    controller.set_line_formation(spacing=1.5)
    positions = controller.get_formation_positions(center, heading)
    for i, pos in enumerate(positions):
        print(f"   Robot {i+1}: ({pos.x:.2f}, {pos.y:.2f})")
    
    print("\nWedge formation:")
    print("-" * 40)
    controller.set_wedge_formation(spacing=2.0)
    positions = controller.get_formation_positions(center, heading)
    for i, pos in enumerate(positions):
        print(f"   Robot {i+1}: ({pos.x:.2f}, {pos.y:.2f})")
    
    print("\nCircle formation:")
    print("-" * 40)
    controller.set_circle_formation(radius=3.0)
    positions = controller.get_formation_positions(center, heading)
    for i, pos in enumerate(positions):
        print(f"   Robot {i+1}: ({pos.x:.2f}, {pos.y:.2f})")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all robotics examples."""
    print("\n" + "=" * 60)
    print("Constraint Theory - Robotics Examples")
    print("=" * 60)
    
    demo_differential_drive()
    demo_robot_arm()
    demo_path_planning()
    demo_sensor_processing()
    demo_formation_control()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways for robotics:

1. Exact Navigation
   - Deterministic heading control
   - Drift-free path following
   - Reproducible robot behavior

2. Kinematics
   - Exact joint angles
   - Precise inverse kinematics
   - Deterministic motion planning

3. Sensor Processing
   - Exact bearing calculations
   - Noise reduction through snapping
   - Deterministic sensor fusion

4. Multi-Robot Coordination
   - Exact formation shapes
   - Coordinated movement
   - Predictable group behavior

For production use:
    from constraint_theory import PythagoreanManifold
    manifold = PythagoreanManifold(200)  # ~1000 exact states
    """)


if __name__ == "__main__":
    main()
