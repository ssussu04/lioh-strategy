"""V5 regression: plan-derived attainment date and both funding regimes."""
import numpy as np
from model import inputs_default, optimize_required_budget, optimize_flexible_funding


def verify_attainment_from_selected_plan(result):
    assert result['feasible'], result['message']
    summary = result['summary']
    deadline = result['deadline']
    required = summary['법률참고목표'].to_numpy(dtype=float)
    effective_threshold = np.maximum(required, float(required[deadline]))
    meets = (summary['비PFE비율'].to_numpy(dtype=float) + 1e-6 >= effective_threshold)
    meets &= summary['미공급_톤'].to_numpy(dtype=float) < 1e-5
    stays = np.logical_and.accumulate(meets[::-1])[::-1]
    valid = np.flatnonzero(stays)
    assert len(valid), 'feasible strict result must support persistent compliance'
    assert valid[0] <= deadline, 'must attain no later than the specified deadline'
    return summary.iloc[int(valid[0])]['월']


if __name__ == '__main__':
    baseline = inputs_default()
    for label, solver in [('uniform', optimize_required_budget), ('variable', optimize_flexible_funding)]:
        result = solver(baseline, time_limit=20)
        date = verify_attainment_from_selected_plan(result)
        print(f'PASS {label}: observed stable start = {date}; optimal={result["optimal"]}; '
              f'total funding MUSD={result["total_funding_MUSD"]:.2f}', flush=True)
