"""Run: python test_v4.py. Checks core constraints and both funding methods."""
import numpy as np
from model import inputs_default, prepare, optimize_flexible_funding, optimize_required_budget

def validate(r, reserve):
    assert r['feasible'], r['message']
    b=r['budgets'];s=r['summary']
    assert (b['월말현금잔액_MUSD'] >= reserve-1e-4).all()
    purchases=r['shipments'].groupby('t')['수입량_톤'].sum().reindex(s['t'],fill_value=0.).to_numpy()
    assert np.max(np.abs(s['수요_톤'].to_numpy()-purchases-s['자체생산_톤'].to_numpy()))<1e-3
    assert s.loc[r['deadline']:,'목표충족'].all()
    assert abs(b['신규예산_MUSD'].sum()+r['data']['config']['초기투자잔액_백만USD']-b['총지출_MUSD'].sum()-b.iloc[-1]['월말현금잔액_MUSD'])<1e-3
    assert (b.loc[~b['월'].str[-2:].isin(['01','04','07','10']),'신규예산_MUSD']<1e-7).all()
    return True

if __name__=='__main__':
    data=inputs_default()
    uniform=optimize_required_budget(data,time_limit=12)
    flex=optimize_flexible_funding(data,time_limit=12)
    for name,r in [('uniform',uniform),('flex',flex)]:
        validate(r,0.)
        print('PASS:',name,'optimal=',r['optimal'],'funding=',round(r['total_funding_MUSD'],3),'ending=',round(r['ending_cash_MUSD'],3),flush=True)
    assert flex['optimal']
    data['config']['최소보유현금_백만USD']=35.714285714
    funded=optimize_flexible_funding(data,time_limit=12)
    validate(funded,35.714285714)
    print('PASS: with minimum cash reserve',flush=True)
    data=inputs_default();data['config']['종료월']='2050-12'
    assert len(prepare(data)['months'])==288
    print('PASS: 2050-12 recognized as 288 months (not a promise of optimal solve)',flush=True)
