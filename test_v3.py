"""Non-network V3 MILP regression tests. Run: python test_v3.py"""
import numpy as np
from model import inputs_default,solve
from scenarios import generate
from decision import decision_gantt
from visuals import replay

def check(result):
    assert result['feasible'],result['message']
    contracts=result['contracts'];ship=result['shipments'];summary=result['summary'];cash=result['budgets']
    assert (cash['월말현금잔액_MUSD'] >= result['data']['config']['최소보유현금_백만USD']-1e-4).all()
    for sid,part in contracts.groupby('id'):
        arr=part.sort_values('t')['계약유지_1'].to_numpy(int)
        starts=part.sort_values('t')['신규계약_1'].to_numpy(int)
        L=int(part['최소계약기간_개월'].iloc[0]); pct=float(part['계약최소구매율'].iloc[0])
        assert (arr>=starts).all()
        for t in np.where(starts==1)[0]:
            assert t+L<=len(arr)
            assert (arr[t:t+L]==1).all()
        prices=result['data']['suppliers'].set_index('id')
        r=prices.loc[sid]; srows=ship.loc[ship['기업'].eq(r['기업'])].sort_values('t')
        monthly=d['supplier_monthly'] if False else result['data']['supplier_monthly']
        for t, row in enumerate(srows.itertuples(index=False)):
            quantity=row.수입량_톤
            # Active contracts must have positive quantity for positive limits / positive min pct.
            assert arr[t] or quantity<1e-5
            if arr[t] and pct>0:assert quantity>1e-6
    if result['optimal']:
        assert (summary['미공급_톤']<1e-6).all()
    return True

if __name__=='__main__':
    d=inputs_default();r=solve(d,time_limit=30);assert r['optimal'];check(r)
    assert len(replay(r).frames)==len(r['summary'])
    assert len(decision_gantt(r).data)>0
    d['config']['최소보유현금_백만USD']=15
    r=solve(d,time_limit=30);assert r['optimal'];check(r)
    dr=generate(42,contract_ranges={'계약기간_난수최소개월':6,'계약기간_난수최대개월':18,'최소구매율_난수최소':.1,'최소구매율_난수최대':.25})
    assert dr['suppliers']['최소계약기간_개월'].between(6,18).all()
    assert dr['suppliers']['계약최소구매율'].between(.1,.25).all()
    dr2=generate(42,contract_ranges={'계약기간_난수최소개월':6,'계약기간_난수최대개월':18,'최소구매율_난수최소':.1,'최소구매율_난수최대':.25})
    assert dr['suppliers'].equals(dr2['suppliers'])
    print('PASS: default MILP, cash reserve, contract commitments, gantt/replay, bounded reproducible random inputs')
