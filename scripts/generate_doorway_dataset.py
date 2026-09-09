import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from ai2thor.controller import Controller
from PIL import Image, ImageDraw

from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate open-doorway renders with YOLO annotations."
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("doorway_dataset"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def door_position(house, door):
    wall = next(wall for wall in house.data["walls"] if wall["id"] == door["wall0"])
    wall_start, wall_end = wall["polygon"][:2]
    wall_length = math.hypot(
        wall_end["x"] - wall_start["x"], wall_end["z"] - wall_start["z"]
    )
    if wall_length == 0:
        raise RuntimeError(f"Door {door['id']} has a zero-length wall.")

    door_min, door_max = door["holePolygon"]
    door_offset = (door_min["x"] + door_max["x"]) / 2
    return (
        {
            "x": wall_start["x"]
            + (wall_end["x"] - wall_start["x"]) * door_offset / wall_length,
            "y": (door_min["y"] + door_max["y"]) / 2,
            "z": wall_start["z"]
            + (wall_end["z"] - wall_start["z"]) * door_offset / wall_length,
        },
        wall_start,
        wall_end,
        wall_length,
    )


def sample_house(controller, random_source):
    for _ in range(25):
        generator = HouseGenerator(
            split="train",
            seed=random_source.randrange(2**32),
            room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
            controller=controller,
        )
        house, _ = generator.sample()
        interior_doors = [
            door
            for door in house.data["doors"]
            if door.get("openable") and door.get("room0") and door.get("room1")
        ]
        if not interior_doors:
            continue

        door = random_source.choice(interior_doors)
        door["openness"] = 1
        if not house.validate(controller):
            return house, door

    raise RuntimeError("Could not generate a navigable house with an openable door.")


def select_camera(controller, create_event, house, door):
    doorway_position, wall_start, wall_end, wall_length = door_position(house, door)
    controller.step(
        action="TeleportFull",
        **house.data["metadata"]["agent"],
        renderImage=False,
    )
    reachable_event = controller.step(action="GetReachablePositions", renderImage=False)
    reachable_positions = reachable_event.metadata["actionReturn"]
    if not reachable_event.metadata["lastActionSuccess"] or not reachable_positions:
        raise RuntimeError(reachable_event.metadata["errorMessage"])

    door_metadata = next(
        obj for obj in create_event.metadata["objects"] if obj["objectId"] == door["id"]
    )
    door_bounds_center = door_metadata["axisAlignedBoundingBox"]["center"]
    wall_normal = {
        "x": -(wall_end["z"] - wall_start["z"]) / wall_length,
        "z": (wall_end["x"] - wall_start["x"]) / wall_length,
    }
    swing_side = math.copysign(
        1,
        (door_bounds_center["x"] - doorway_position["x"]) * wall_normal["x"]
        + (door_bounds_center["z"] - doorway_position["z"]) * wall_normal["z"],
    )
    clear_side = -swing_side
    wall_tangent = {
        "x": (wall_end["x"] - wall_start["x"]) / wall_length,
        "z": (wall_end["z"] - wall_start["z"]) / wall_length,
    }

    candidates = []
    for position in reachable_positions:
        relative_x = position["x"] - doorway_position["x"]
        relative_z = position["z"] - doorway_position["z"]
        distance = math.hypot(relative_x, relative_z)
        normal_distance = clear_side * (
            relative_x * wall_normal["x"] + relative_z * wall_normal["z"]
        )
        lateral_distance = abs(
            relative_x * wall_tangent["x"] + relative_z * wall_tangent["z"]
        )
        if 1.2 <= distance <= 3.0 and normal_distance >= 1.0:
            candidates.append((position, normal_distance, lateral_distance))

    if not candidates:
        raise RuntimeError("No clear camera position was found facing the selected door.")

    camera_position, normal_distance, lateral_distance = min(
        candidates,
        key=lambda candidate: (
            abs(candidate[1] - 2.0),
            candidate[2],
        ),
    )
    horizontal_distance = math.hypot(
        doorway_position["x"] - camera_position["x"],
        doorway_position["z"] - camera_position["z"],
    )
    camera_height = 1.5
    camera_rotation = {
        "x": math.degrees(
            math.atan2(camera_height - doorway_position["y"], horizontal_distance)
        ),
        "y": math.degrees(
            math.atan2(
                doorway_position["x"] - camera_position["x"],
                doorway_position["z"] - camera_position["z"],
            )
        ),
        "z": 0,
    }
    return camera_position, camera_height, camera_rotation


def save_annotations(mask, output_dir, image_stem):
    height, width = mask.shape
    y_coordinates, x_coordinates = np.where(mask)
    if not len(x_coordinates):
        raise RuntimeError("The selected doorway is not visible in the render.")

    x_min, x_max = x_coordinates.min(), x_coordinates.max()
    y_min, y_max = y_coordinates.min(), y_coordinates.max()
    box = (
        (x_min + x_max + 1) / (2 * width),
        (y_min + y_max + 1) / (2 * height),
        (x_max - x_min + 1) / width,
        (y_max - y_min + 1) / height,
    )
    (output_dir / "boxes" / f"{image_stem}.txt").write_text(
        "0 " + " ".join(f"{value:.6f}" for value in box) + "\n"
    )

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
    polygon = [
        coordinate
        for point in contour
        for coordinate in (point[0] / width, point[1] / height)
    ]
    if len(polygon) < 6:
        raise RuntimeError("The doorway mask does not contain a valid polygon.")
    (output_dir / "labels" / f"{image_stem}.txt").write_text(
        "0 " + " ".join(f"{value:.6f}" for value in polygon) + "\n"
    )
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        output_dir / "masks" / f"{image_stem}.png"
    )
    return x_min, y_min, x_max, y_max


def save_overlay(image, mask, bounding_box, destination):
    image_array = np.asarray(image).copy()
    image_array[mask] = (0.65 * image_array[mask] + 0.35 * np.array([0, 255, 0])).astype(
        np.uint8
    )
    overlay = Image.fromarray(image_array)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bounding_box, outline="red", width=3)
    overlay.save(destination)


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")

    output_dir = args.output_dir
    for directory in ("images", "labels", "boxes", "masks", "overlays"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    (output_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: open_doorway\n"
    )

    random_source = random.Random(args.seed)
    controller = Controller(
        width=640,
        height=640,
        quality="Low",
        renderInstanceSegmentation=True,
        **PROCTHOR_INITIALIZATION,
    )
    try:
        generated = 0
        attempts = 0
        while generated < args.count:
            attempts += 1
            if attempts > args.count * 10:
                raise RuntimeError("Could not find enough renderable open doorways.")

            try:
                house, door = sample_house(controller, random_source)
                controller.reset(renderImage=True, renderInstanceSegmentation=True)
                create_event = controller.step(
                    action="CreateHouse", house=house.data, renderImage=True
                )
                camera_position, camera_height, camera_rotation = select_camera(
                    controller, create_event, house, door
                )
                render_event = controller.step(
                    action="AddThirdPartyCamera",
                    position={
                        "x": camera_position["x"],
                        "y": camera_height,
                        "z": camera_position["z"],
                    },
                    rotation=camera_rotation,
                    fieldOfView=80,
                    renderImage=True,
                )
                mask = render_event.third_party_instance_masks[0].get(door["id"])
                if mask is None:
                    continue

                image_stem = f"doorway_{args.start_index + generated:03d}"
                image = Image.fromarray(render_event.third_party_camera_frames[0][..., :3])
                bounding_box = save_annotations(mask, output_dir, image_stem)
                image.save(output_dir / "images" / f"{image_stem}.png")
                save_overlay(
                    image,
                    mask,
                    bounding_box,
                    output_dir / "overlays" / f"{image_stem}.png",
                )
                generated += 1
                print(f"Generated {generated}/{args.count}: {image_stem}")
            except RuntimeError as error:
                print(f"Skipping candidate: {error}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
