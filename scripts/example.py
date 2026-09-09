import math
import random

from PIL import Image

from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

house_generator = HouseGenerator(
    split="train",
    seed=random.randrange(2**32),
    room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
)
for _ in range(10):
    house, _ = house_generator.sample()
    interior_doors = [
        door
        for door in house.data["doors"]
        if door.get("openable") and door.get("room0") and door.get("room1")
    ]
    if not interior_doors:
        continue

    door = random.choice(interior_doors)
    door["openness"] = 1
    if not house.validate(house_generator.controller):
        break
else:
    raise RuntimeError("Could not generate a navigable house with an openable interior door.")

wall = next(wall for wall in house.data["walls"] if wall["id"] == door["wall0"])
wall_start, wall_end = wall["polygon"][:2]
wall_length = math.hypot(
    wall_end["x"] - wall_start["x"], wall_end["z"] - wall_start["z"]
)
if wall_length == 0:
    raise RuntimeError(f"Selected door {door['id']} has a zero-length wall.")

door_min, door_max = door["holePolygon"]
door_offset = (door_min["x"] + door_max["x"]) / 2
door_position = {
    "x": wall_start["x"]
    + (wall_end["x"] - wall_start["x"]) * door_offset / wall_length,
    "y": (door_min["y"] + door_max["y"]) / 2,
    "z": wall_start["z"]
    + (wall_end["z"] - wall_start["z"]) * door_offset / wall_length,
}

house_generator.controller.reset(renderImage=True)
create_event = house_generator.controller.step(
    action="CreateHouse",
    house=house.data,
    renderImage=True,
)
house_generator.controller.step(
    action="TeleportFull",
    **house.data["metadata"]["agent"],
    renderImage=False,
)

reachable_event = house_generator.controller.step(
    action="GetReachablePositions", renderImage=False
)
reachable_positions = reachable_event.metadata["actionReturn"]
if not reachable_event.metadata["lastActionSuccess"] or not reachable_positions:
    raise RuntimeError(
        "AI2-THOR could not find reachable camera positions: "
        f"{reachable_event.metadata['errorMessage']}"
    )
door_metadata = next(
    obj for obj in create_event.metadata["objects"] if obj["objectId"] == door["id"]
)
door_bounds_center = door_metadata["axisAlignedBoundingBox"]["center"]
wall_normal = {
    "x": -(wall_end["z"] - wall_start["z"]) / wall_length,
    "z": (wall_end["x"] - wall_start["x"]) / wall_length,
}
door_swing_side = math.copysign(
    1,
    (door_bounds_center["x"] - door_position["x"]) * wall_normal["x"]
    + (door_bounds_center["z"] - door_position["z"]) * wall_normal["z"],
)
clear_side = -door_swing_side

def camera_scores(position):
    relative_x = position["x"] - door_position["x"]
    relative_z = position["z"] - door_position["z"]
    distance = math.hypot(relative_x, relative_z)
    normal_distance = clear_side * (
        relative_x * wall_normal["x"] + relative_z * wall_normal["z"]
    )
    lateral_distance = abs(
        relative_x * (wall_end["x"] - wall_start["x"]) / wall_length
        + relative_z * (wall_end["z"] - wall_start["z"]) / wall_length
    )
    return distance, normal_distance, lateral_distance

nearby_positions = [
    position
    for position in reachable_positions
    if 1.2 <= camera_scores(position)[0] <= 3.0
    and camera_scores(position)[1] >= 1.0
]
if not nearby_positions:
    raise RuntimeError("No clear camera position was found facing the selected door.")

camera_position = min(
    nearby_positions,
    key=lambda position: (
        abs(camera_scores(position)[1] - 2.0),
        camera_scores(position)[2],
    ),
)
horizontal_distance = math.hypot(
    door_position["x"] - camera_position["x"],
    door_position["z"] - camera_position["z"],
)
camera_height = 1.5
camera_rotation = {
    "x": math.degrees(
        math.atan2(camera_height - door_position["y"], horizontal_distance)
    ),
    "y": math.degrees(
        math.atan2(
            door_position["x"] - camera_position["x"],
            door_position["z"] - camera_position["z"],
        )
    ),
    "z": 0,
}
house_generator.controller.step(
    action="AddThirdPartyCamera",
    position={"x": camera_position["x"], "y": camera_height, "z": camera_position["z"]},
    rotation=camera_rotation,
    fieldOfView=80,
)
Image.fromarray(house_generator.controller.last_event.third_party_camera_frames[0]).save(
    "house.png"
)

house.to_json("temp.json")
