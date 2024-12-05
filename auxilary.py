from arcade import Sprite

class Rescuer(Sprite):
    def __init__(self, *args, health: float = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.health: float = health
        self.carries_resource: bool = False
        self.resource_carried: Resource | None


class Resource(Sprite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_stuck: bool = False

    def update(self):
        if not self.stuck:
            self.center_x += self.change_x
            self.center_y += self.change_y


class Action():
    def __init__(self, action_type: str, sprite: Sprite) -> None:
        self.action_type: str = action_type
        self.sprite: Sprite = sprite


class Move(Action):
    def __init__(self, action_type: str, sprite: Sprite, force: tuple) -> None:
        super().__init__(action_type, sprite)
        self.force = force
