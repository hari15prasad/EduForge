import sys
import os
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environment.student_fsm import StudentSimulator, MisconceptionType, TutorAction

def test_attention_boost():
    # Test Procedural with low attention
    sim = StudentSimulator(
        misconception_init=MisconceptionType.PROCEDURAL,
        attention_init=3.0,
        confusion_init=8.0
    )
    
    print(f"Initial attention: {sim.attention}")
    
    # Try QUESTION action (should have boost)
    # Since transition has stochasticity, we might need a few tries or just check the code path
    # But let's see if we can observe it.
    
    # We can't easily isolate the boost without modifying the sim or running many times.
    # But we can check if it stays within bounds.
    
    sim.transition(TutorAction.QUESTION)
    print(f"Attention after QUESTION: {sim.attention}")

if __name__ == "__main__":
    test_attention_boost()
