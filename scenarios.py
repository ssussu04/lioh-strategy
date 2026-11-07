"""Reproducible research-assumption generator. NEVER legal eligibility randomization."""
import copy
import numpy as np
import pandas as pd
from model import inputs_default, clone

RANGE_INFO = {
    '월 기초수요': '기준 1,250톤의 0.8~1.4배, 연 성장률 0~6% (가정)',
    '월별 수요 변동': '계절성 최대 ±7%, 일시적 수요 충격 ±5% (가정)',
    '공장 CAPEX': '국가별 조사값의 0.75~1.30배 (가정)',
    '공장 변동비·고정비': '조사값의 0.75~1.30배 (가정)',
    '공장 연 생산능력': '기준 25,000톤의 0.65~1.25배 (가정)',
    '공장 실효 가동·정제 수율': '합산 실효율 70~98% (가정)',
    '공장 원료 확보': '설계 월 생산량의 약 35~130%를 충족하는 SC6 양 (가정)',
    '공장 건설기간': '36~72개월 (가정)',
    '공급기업 단가': '국가별 수입단가 대리값의 0.75~1.30배 (가정)',
    '공급기업 계약상한': '확인된 기존 상한의 0.65~1.45배 (가정)',
    '월별 공급 차질': '보통 0.65~1.10배, 일부 달에 0.1~0.6배 공급차질 (가정)',
    '최소 계약기간': '랜덤 범위 기본 24~72개월, 개별 수정 가능',
    '계약 최소구매율': '랜덤 범위 기본 5~25%, 개별 수정 가능',
    '공급계약 중단': '활성 기업별 확률적 종료년도 2030~2055 (시나리오 가정)',
    '분기별 CAPEX 예산': '균등 분기 신규자금 0~90백만 USD (가정)',
}

def generate(seed: int, intensity: str='표준', keep_budget: bool=False, contract_ranges=None):
    """Return filled editable inputs. Base legal classifications and zero unknown caps remain unchanged."""
    d=inputs_default();rng=np.random.default_rng(int(seed));cfg=d['config']
    if contract_ranges:cfg.update(contract_ranges)
    p={'보수적':0.45,'표준':1.0,'스트레스':1.4}.get(intensity,1.0)
    def factor(low,high,n=None):
        a=1+(low-1)*p;b=1+(high-1)*p
        return rng.uniform(max(.02,a),b,n)
    suppliers=d['suppliers'];plants=d['plants'];mo=d['months']
    suppliers['단가_USD톤']=(suppliers['단가_USD톤']*factor(.75,1.30,len(suppliers))).round(1)
    suppliers['연공급상한_톤']=(suppliers['연공급상한_톤']*factor(.65,1.45,len(suppliers))).round(1)
    suppliers['운송비_USD톤']=(rng.uniform(30,470,len(suppliers))*p).round(1)
    lo=int(cfg.get('계약기간_난수최소개월',24)); hi=int(cfg.get('계약기간_난수최대개월',72))
    p_lo=float(cfg.get('최소구매율_난수최소',.05));p_hi=float(cfg.get('최소구매율_난수최대',.25))
    if not (1<=lo<=hi<=360):raise ValueError('난수 최소계약기간은 1~360개월, 최소값은 최대값 이하여야 합니다')
    if not (0<=p_lo<=p_hi<=1):raise ValueError('최소구매율 난수범위는 0~100%입니다')
    suppliers['최소계약기간_개월']=rng.integers(lo,hi+1,len(suppliers))
    suppliers['계약최소구매율']=rng.uniform(p_lo,p_hi,len(suppliers)).round(3)
    suppliers['월변경상한_톤']=[round(float(v)/12*rng.uniform(.25,.75),1) if v>0 and rng.random()<.65 else 0. for v in suppliers['연공급상한_톤']]
    for idx,r in suppliers.iterrows():
        if r['연공급상한_톤']>0 and rng.random()<.30*p:
            # Integer-year commitment and scheduled contract expiry, not a random PFE reclassification.
            stop_year=int(rng.integers(2030,2056))
            suppliers.at[idx,'공급중단시작월']=f'{stop_year}-01'
        elif r['연공급상한_톤']>0:
            suppliers.at[idx,'공급중단시작월']=''
    plants['건설비_백만USD']=(plants['건설비_백만USD']*factor(.75,1.30,len(plants))).round(2)
    plants['변동비_USD톤']=(plants['변동비_USD톤']*factor(.75,1.30,len(plants))).round(1)
    plants['고정비_백만USD년']=(plants['고정비_백만USD년']*factor(.75,1.30,len(plants))).round(2)
    plants['건설기간_개월']=rng.integers(36,73,len(plants))
    plants['초기가동률']=rng.uniform(.20,.60,len(plants)).round(2)
    plants['증산기간_개월']=rng.integers(4,13,len(plants))
    plants['운송비_USD톤']=(rng.uniform(40,640,len(plants))*p).round(1)
    plants['설비실효율']=(rng.uniform(.70,.98,len(plants))).round(3)
    plants['생산능력_톤년']=(plants['생산능력_톤년']*factor(.65,1.25,len(plants))).round(0)
    # Plant feedstock is now an actual potential bottleneck rather than 3x nameplate.
    plants['원료확보_SC6톤월']=(plants['생산능력_톤년']/12*plants['원료투입계수']*rng.uniform(.35,1.30,len(plants))).round(0)
    annual_growth=float(rng.uniform(0,.06*p));initial_d=float(1250*factor(.8,1.4))
    n=len(mo);yrs=np.arange(n)/12
    season=1+rng.uniform(0,.07*p)*np.sin(2*np.pi*np.arange(n)/12+rng.uniform(0,6.28))
    shocks=rng.uniform(1-.05*p,1+.05*p,n)
    mo['수요_톤']=np.maximum(0,initial_d*(1+annual_growth)**yrs*season*shocks).round(1)
    budget=float(rng.uniform(0,90)) if not keep_budget else 50.
    mo['신규투자예산_백만USD']=[budget if x[-2:] in ['01','04','07','10'] else 0. for x in mo['월']]
    sm=d['supplier_monthly'];pm=d['plant_monthly']
    for company in suppliers['id']:
        mask=sm['id'].eq(company)
        prices=np.clip(rng.normal(1,.055*p,mask.sum()),.75,1.35)
        capacities=np.clip(rng.normal(1,.13*p,mask.sum()),.5,1.2)
        shocks_idx=rng.random(mask.sum())<(.025*p)
        capacities[shocks_idx]*=rng.uniform(.1,.6,shocks_idx.sum())
        sm.loc[mask,'가격배율']=prices.round(3)
        sm.loc[mask,'공급배율']=capacities.round(3)
    for country in plants['국가']:
        mask=pm['국가'].eq(country)
        values=np.clip(rng.normal(1,.12*p,mask.sum()),.4,1.25)
        shock_idx=rng.random(mask.sum())<(.025*p)
        values[shock_idx]*=rng.uniform(.2,.7,shock_idx.sum())
        pm.loc[mask,'원료배율']=values.round(3)
    cfg.update({'난수사용':False,'난수시드':int(seed),'생성시드':int(seed),'생성강도':intensity,'연수요성장률_가정':round(annual_growth,4), '생성방식':'조사값 기준 시나리오 추정 (법적 비PFE 분류 미변경)'})
    return d
