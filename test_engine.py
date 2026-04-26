import unittest
from src.rewards.engine import RewardEngine, SCAFFOLD_ACTIONS, DIRECT_TELL_ACTIONS, MIN_DIAGNOSTIC_DEPTH, EARLY_SESSION_THRESHOLD
from src.environment.student_fsm import TutorAction, MisconceptionType

class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RewardEngine()

    def test_info_dump_penalty_early_session(self):
        # A direct tell (worked_example) on step 1 should get a heavy penalty
        total, components = self.engine.compute(
            confusion_before=8.0, confusion_after=6.0,
            attention_after=6.0,
            action=TutorAction.WORKED_EXAMPLE,
            episode_length=1,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        # penalty = 2.0 * max(0, 4 - 1 + 1) = 2.0 * 4 = 8.0
        self.assertEqual(components.breakdown["p_info_dump"], 8.0)

    def test_info_dump_no_penalty_late_session(self):
        # A direct tell on step 6 (past early session threshold) should have 0 penalty
        total, components = self.engine.compute(
            confusion_before=8.0, confusion_after=6.0,
            attention_after=6.0,
            action=TutorAction.WORKED_EXAMPLE,
            episode_length=6,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        self.assertEqual(components.breakdown["p_info_dump"], 0.0)

    def test_scaffold_confusion_drop_higher_reward(self):
        # Dropping confusion by 2.0 using a scaffold action
        _, comp_scaffold = self.engine.compute(
            confusion_before=8.0, confusion_after=6.0, attention_after=6.0,
            action=TutorAction.QUESTION, episode_length=3,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        # Dropping confusion by 2.0 using a direct tell
        _, comp_direct = self.engine.compute(
            confusion_before=8.0, confusion_after=6.0, attention_after=6.0,
            action=TutorAction.WORKED_EXAMPLE, episode_length=3,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        # Process reward for scaffold should be higher than direct
        # scaffold: 2.0 * 1.4 * 1.0 * 2.0 = 5.6
        # direct: 2.0 * 0.6 * 1.0 * 2.0 = 2.4
        self.assertGreater(comp_scaffold.r_process, comp_direct.r_process)
        self.assertAlmostEqual(comp_scaffold.r_process, 5.6)
        self.assertAlmostEqual(comp_direct.r_process, 2.4)

    def test_alignment_priority_tiered(self):
        # For PROCEDURAL, hint > question > worked_example
        # hint is 0th (2.5), question is 1st (1.5), worked_example is 2nd (0.5)
        _, comp_hint = self.engine.compute(
            confusion_before=8.0, confusion_after=8.0, attention_after=6.0,
            action=TutorAction.HINT, episode_length=3,
            misconception=MisconceptionType.PROCEDURAL,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        self.assertEqual(comp_hint.r_alignment, 2.5)

        _, comp_wex = self.engine.compute(
            confusion_before=8.0, confusion_after=8.0, attention_after=6.0,
            action=TutorAction.WORKED_EXAMPLE, episode_length=3,
            misconception=MisconceptionType.PROCEDURAL,
            action_text="test", format_valid=True, done=False, done_reason=None
        )
        self.assertEqual(comp_wex.r_alignment, 0.5)

    def test_scaffolded_resolution_bonus_requires_depth(self):
        # If resolving in 3 steps, no depth bonus should be given even if scaffolded
        self.engine._scaffold_streak = 3
        _, comp_early = self.engine.compute(
            confusion_before=2.0, confusion_after=1.0, attention_after=6.0,
            action=TutorAction.QUESTION, episode_length=3,
            action_text="test", format_valid=True, done=True, done_reason="success"
        )
        # Early completion -> no scaffolded resolution bonus
        self.assertEqual(comp_early.breakdown["r_scaffolded_resolution"], 0.0)

        # If resolving in 6 steps (>= MIN_DIAGNOSTIC_DEPTH), bonus is applied
        self.engine._scaffold_streak = 3
        _, comp_valid = self.engine.compute(
            confusion_before=2.0, confusion_after=1.0, attention_after=6.0,
            action=TutorAction.QUESTION, episode_length=6,
            action_text="test", format_valid=True, done=True, done_reason="success"
        )
        self.assertEqual(comp_valid.breakdown["r_scaffolded_resolution"], 8.0)

if __name__ == '__main__':
    unittest.main()
