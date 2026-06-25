"""
cluster.py — GPU, Rack, Cluster models for watt-bench.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

PROFILES_PATH = Path(__file__).parent.parent / "hardware_profiles.json"


def _load_profiles():
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _nearest_batch(batch_size: int) -> str:
    keys = [1, 8, 32, 64]
    nearest = min(keys, key=lambda k: abs(k - batch_size))
    return f"batch_{nearest}"


@dataclass
class Job:
    id: str
    model: str
    prompt_tokens: int
    output_tokens: int
    sla_latency_ms: float
    arrival_time: float
    batch_size: int = 1
    assigned_gpu: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    throttled: bool = False

    @property
    def latency_ms(self):
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def sla_violated(self):
        lat = self.latency_ms
        if lat is None:
            return False
        return lat > self.sla_latency_ms


@dataclass
class GPU:
    id: str
    gpu_type: str
    rack_id: str
    profile: dict
    current_jobs: list = field(default_factory=list)
    is_throttled: bool = False

    def vram_used(self):
        used = 0.0
        for j in self.current_jobs:
            mp = self.profile["models"].get(j.model, {})
            used += mp.get("vram_required_gb", 0)
        return used

    def can_fit(self, job: Job) -> bool:
        mp = self.profile["models"].get(job.model)
        if not mp:
            return False
        return (self.profile["vram_gb"] - self.vram_used()) >= mp["vram_required_gb"]

    def tok_per_sec(self, job: Job) -> float:
        mp = self.profile["models"].get(job.model, {})
        bk = _nearest_batch(job.batch_size)
        base = mp.get(bk, {}).get("tok_s", 0.0)
        if self.is_throttled:
            factor = self.profile.get("throttle_factor", 0.70)
            return base * factor
        return base

    def current_power_w(self) -> float:
        if not self.current_jobs:
            return self.profile["idle_w"]
        j = self.current_jobs[0]
        mp = self.profile["models"].get(j.model, {})
        bk = _nearest_batch(j.batch_size)
        base = mp.get(bk, {}).get("power_w", self.profile["tdp_w"])
        if self.is_throttled:
            # throttle reduces power to threshold level
            return self.profile.get("throttle_threshold_w", base * 0.65)
        return base


@dataclass
class Rack:
    id: str
    power_budget_w: float
    cooling_limit_w: float
    gpus: List[GPU]
    throttle_events: int = 0

    def total_power_w(self) -> float:
        return sum(g.current_power_w() for g in self.gpus)

    def headroom_w(self) -> float:
        return self.power_budget_w - self.total_power_w()

    def apply_thermal_check(self) -> bool:
        if self.total_power_w() > self.cooling_limit_w:
            for g in self.gpus:
                g.is_throttled = True
            self.throttle_events += 1
            return True
        else:
            for g in self.gpus:
                g.is_throttled = False
            return False

    def available_gpus(self, job: Job) -> List[GPU]:
        return [g for g in self.gpus if g.can_fit(job)]


class Cluster:
    def __init__(self, name: str, racks: List[Rack]):
        self.name = name
        self.racks = racks

    @classmethod
    def from_preset(cls, preset_name: str) -> "Cluster":
        path = Path(__file__).parent / "presets" / f"{preset_name}.json"
        with open(path) as f:
            cfg = json.load(f)
        profiles = _load_profiles()
        racks = []
        for rc in cfg["racks"]:
            gpus = []
            for gc in rc["gpus"]:
                gpus.append(GPU(
                    id=gc["id"],
                    gpu_type=gc["type"],
                    rack_id=rc["id"],
                    profile=profiles[gc["type"]],
                ))
            racks.append(Rack(
                id=rc["id"],
                power_budget_w=rc["power_budget_w"],
                cooling_limit_w=rc.get("cooling_limit_w") or rc.get("cooling_dissipation_w", rc["power_budget_w"] * 0.95),
                gpus=gpus,
            ))
        return cls(name=cfg["name"], racks=racks)

    def all_gpus(self) -> List[GPU]:
        return [g for r in self.racks for g in r.gpus]

    def get_gpu(self, gpu_id: str) -> Optional[GPU]:
        for g in self.all_gpus():
            if g.id == gpu_id:
                return g
        return None

    def get_rack_for_gpu(self, gpu_id: str) -> Optional[Rack]:
        for r in self.racks:
            for g in r.gpus:
                if g.id == gpu_id:
                    return r
        return None

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_gpus": len(self.all_gpus()),
            "total_racks": len(self.racks),
            "total_power_budget_w": sum(r.power_budget_w for r in self.racks),
        }
