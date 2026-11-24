# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Retain measured stage costs and disclose missing wall time after interrupted sessions."""

import math


def account_upfront_cost(variant, training, merging, ledger):
    """Use per-stage component lower bounds when a stage wall ledger is incomplete."""
    name = variant['id']
    stages = [(f'train_{name}', sum(training[key] for key in [
        'student_rollout_seconds', 'teacher_scoring_seconds', 'student_backward_and_optimizer_seconds'])),
        (f'merge_{name}', merging['merge_compute_seconds'] + merging['merge_write_and_backup_seconds'])]
    if variant['method'] == 'off_policy':
        stages.append((f"pool_seed{variant['seed']}", training['teacher_pool_generation_seconds']))
    recorded_total, accounted_total, component_total, missing = 0., 0., 0., []
    for stage, component in stages:
        entries = [row for row in ledger if row['stage'] == stage]
        observed = sum(row['seconds'] for row in entries)
        if not all(math.isfinite(value) and value >= 0 for value in [observed, component]):
            raise ValueError('An upfront execution cost is non-finite or negative.')
        recorded_total += observed
        component_total += component
        accounted_total += max(observed, component)
        if not entries or component > observed + 1e-6:
            missing.append(stage)
    return {'observed_upfront_wall_seconds': recorded_total,
        'measured_component_seconds': component_total,
        'accounted_upfront_seconds': accounted_total,
        'upfront_is_lower_bound': bool(missing),
        'incomplete_wall_stages': ';'.join(missing),
        'cost_scope': 'sum over pool/training/merge of max(recorded stage wall, full-history measured components); '
            'a missing stage interval yields a lower bound, not a reconstructed exact wall time'}
