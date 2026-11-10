from __future__ import annotations

import json
import zipfile

ALGOS = ("ppo", "sac")


def algo_class(algo):
    """Get the stable-baselines3 class of an algorithm name.

    Args:
        algo (str): Algorithm name, ppo or sac

    Returns:
        type: PPO or SAC class
    """
    name = str(algo).lower()
    if name == "ppo":
        from stable_baselines3 import PPO
        return PPO
    if name == "sac":
        from stable_baselines3 import SAC
        return SAC
    raise ValueError(f"unknown algo {algo!r}; expected one of {ALGOS}")


def algo_of(path):
    """Find which algorithm saved a model zip.

    Args:
        path (str): Path of the model zip

    Returns:
        str: ppo or sac
    """
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path} is not a stable-baselines3 model zip: {exc}") from exc
    cls = data.get("policy_class") or {}
    blob = f"{cls.get('__module__', '')} {cls.get(':serialized:', '')}".lower()
    if "sac" in blob or "buffer_size" in data:
        return "sac"
    return "ppo"


def load_policy(path, **kwargs):
    """Load a saved policy with the class that saved it.

    Args:
        path (str): Path of the model zip
        **kwargs (dict): Extra arguments for load

    Returns:
        object: The loaded model
    """
    return algo_class(algo_of(path)).load(path, **kwargs)


def _selfcheck():
    """Save and reload a PPO and a SAC model and check the detected algorithm.

    Returns:
        bool: True if every check passes
    """
    import tempfile, os
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for name, cls in (("ppo", PPO), ("sac", SAC)):
            m = cls("MlpPolicy", gym.make("Pendulum-v1"), **({} if name == "ppo" else
                                                            {"buffer_size": 1000}))
            p = os.path.join(d, f"{name}.zip")
            m.save(p)
            got = algo_of(p)
            print(f"  {'ok ' if got == name else 'FAIL'} algo_of({name}.zip) = {got}")
            ok &= got == name
            back = load_policy(p, device="cpu")
            print(f"  {'ok ' if isinstance(back, cls) else 'FAIL'} load_policy -> {type(back).__name__}")
            ok &= isinstance(back, cls)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selfcheck() else 1)
