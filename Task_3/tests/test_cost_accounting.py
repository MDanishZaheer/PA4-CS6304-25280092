# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check interruption-aware cost accounting without rerunning scientific experiments."""

import unittest

from Task_3.analysis.costs import account_upfront_cost


class CostAccountingTests(unittest.TestCase):
    def setUp(self):
        self.variant = {'id': 'on_policy_seed6305', 'method': 'on_policy', 'seed': 6305}
        self.training = {'teacher_pool_generation_seconds': 0., 'student_rollout_seconds': 1800.,
                         'teacher_scoring_seconds': 50., 'student_backward_and_optimizer_seconds': 190.}
        self.merging = {'merge_compute_seconds': 1., 'merge_write_and_backup_seconds': 12.}

    def test_resumed_stage_cannot_erase_completed_training_cost(self):
        ledger = [{'stage': 'train_on_policy_seed6305', 'seconds': 175.},
                  {'stage': 'merge_on_policy_seed6305', 'seconds': 22.}]
        row = account_upfront_cost(self.variant, self.training, self.merging, ledger)
        self.assertEqual(row['observed_upfront_wall_seconds'], 197.)
        self.assertEqual(row['accounted_upfront_seconds'], 2062.)
        self.assertTrue(row['upfront_is_lower_bound'])
        self.assertEqual(row['incomplete_wall_stages'], 'train_on_policy_seed6305')

    def test_complete_stages_keep_actual_wall_cost_and_pool_preparation(self):
        self.variant.update(id='off_policy_seed6304', method='off_policy', seed=6304)
        self.training.update(teacher_pool_generation_seconds=3400., student_rollout_seconds=0.)
        ledger = [{'stage': 'train_off_policy_seed6304', 'seconds': 270.},
                  {'stage': 'pool_seed6304', 'seconds': 3450.},
                  {'stage': 'merge_off_policy_seed6304', 'seconds': 22.}]
        row = account_upfront_cost(self.variant, self.training, self.merging, ledger)
        self.assertEqual(row['accounted_upfront_seconds'], 3742.)
        self.assertFalse(row['upfront_is_lower_bound'])


if __name__ == '__main__':
    unittest.main()
