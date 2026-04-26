"""
run_env_demo.py — Simple runner to test EduForgeEnv directly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.openenv_wrapper import EduForgeEnv


def main():
    print("\n" + "=" * 70)
    print("EduForge Environment Demo")
    print("=" * 70 + "\n")

    env = EduForgeEnv(seed=42)
    obs = env.reset()

    print(f"Misconception : {obs.misconception_id.value}")
    print(f"Confusion     : {obs.confusion}")
    print(f"Attention     : {obs.attention}")
    print(f"Student       : '{obs.student_response}'")
    print()

    actions = [
        "<STRATEGY>explain</STRATEGY>",
        "<STRATEGY>worked_example</STRATEGY>",
        "<STRATEGY>question</STRATEGY>",
        "<STRATEGY>analogize</STRATEGY>",
        "<STRATEGY>correct_fact</STRATEGY>",
    ]

    for i, action in enumerate(actions, 1):
        obs, reward, done, info = env.step(action)

        print(f"Step {i}")
        print(f"  Strategy  : {info.parsed_action.value if info.parsed_action else 'UNPARSED'}")
        print(f"  Confusion : {obs.confusion:.2f}")
        print(f"  Attention : {obs.attention:.2f}")
        print(f"  Reward    : {reward:+.4f}")
        print(f"  Student   : '{obs.student_response}'")
        print(f"  Done      : {done}")
        if done:
            print(f"  Reason    : {info.done_reason}")
        print()

        if done:
            break

    print("=" * 70)
    print(f"Episode ended after {env.turn_count} turns")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
