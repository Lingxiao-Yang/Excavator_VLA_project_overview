"""Episode success detector used for the 200-episode evaluation.

Success = new particles from the initial pile are inside the truck-bed volume, that
count has stopped growing for 3 consecutive steps, and the bucket is open (dumped).
The report is explicit that this is *deliberately permissive* (>= 1 particle); it does
not measure payload or spillage.
"""

from dataclasses import dataclass


@dataclass
class SuccessDetector:
    min_new_particles: int = 1
    stable_steps: int = 3
    min_bucket_open_rad: float = -0.80
    _last: int = -1
    _stable: int = 0

    def update(self, truck_new_particles: int, bucket_angle_rad: float) -> bool:
        if truck_new_particles == self._last:
            self._stable += 1
        else:
            self._stable, self._last = 0, truck_new_particles
        return (truck_new_particles >= self.min_new_particles
                and self._stable >= self.stable_steps
                and bucket_angle_rad >= self.min_bucket_open_rad)
