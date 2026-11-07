"""LiOH supply chain decision support MILP. Monthly flows; quarterly enterprise cash funding for procurement, CAPEX and OPEX.
All eligibility is a *research proxy*, NOT a 45X statutory MACR determination.
All monetary objectives use USD millions for numerical stability.
"""
from __future__ import annotations
import copy, io, json, math
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint, linprog
from scipy.sparse import coo_matrix

SUPPLIERS=[
 ('CN-GAN','중국','Ganfeng Lithium',0,16090,15187.5,'2024 국가별 수입단가 대리값'),
 ('CN-CHX','중국','Shenzhen Chengxin Lithium',0,16090,0,'한국향 공급상한 미확정'),
 ('CL-SQM','칠레','SQM',0,14000,7500,'비PFE 여부 보수적으로 0 가정'),
 ('AU-COV','호주','Covalent Lithium',0,11580,4375,'비PFE 여부 보수적으로 0 가정'),
 ('AU-TLEA','호주','TLEA',0,11580,2209.75,'Tianqi 51% 지분 관련 가정'),
 ('US-RIO','미국','Rio Tinto',1,22090,2812.5,'비PFE 잠정 가정'),
 ('AU-ALB','호주','Albemarle Kemerton',1,11580,0,'가동중단 발표, 가용 물량 0 가정'),
 ('AU-AST','호주','Asteron Lithium Processing',1,14900,800,'scenario_assumption'),
 ('CL-MER','칠레','Meridian Brine Refining',1,16200,900,'scenario_assumption'),
 ('US-SOL','미국','Solvanta Materials',1,18600,1100,'scenario_assumption'),
 ('CA-BLU','캐나다','Bluecrest Lithium Chemicals',1,17700,900,'scenario_assumption'),
 ('AR-PAC','아르헨티나','Pacifica Hydroxide Partners',0,13600,7200,'scenario_assumption')]
PLANTS=[('칠레',492,48,25000,19689,19.592),('호주',547,48,25000,20674,21.769),('미국',800,48,25000,22151,25.034)]
REG_MIN={2027:0,2028:0,2029:0,2030:.25,2031:.30,2032:.40,2033:.50,2034:.50,2035:.50,2036:.50,2037:.50,2038:.50,2039:.50,2040:.50}
REG_BAT={2027:.65,2028:.70,2029:.80,2030:.85,2031:.85,2032:.85,2033:.85,2034:.85,2035:.85}

def inputs_default():
    months=pd.date_range('2027-01-01','2059-12-01',freq='MS')
    suppliers=pd.DataFrame(SUPPLIERS,columns=['id','국가','기업','비PFE_1','단가_USD톤','연공급상한_톤','출처_주의'])
    suppliers['운송비_USD톤']=0.;suppliers['평가재료가치_USD톤']=15000.;suppliers['월변경상한_톤']=0.;suppliers['사용_1']=1
    suppliers['공급시작월']='2027-01';suppliers['공급중단시작월']=''
    suppliers['최소계약기간_개월']=48
    suppliers['계약최소구매율']=0.15
    suppliers.loc[suppliers['id']=='AU-AST','공급중단시작월']='2035-01'
    suppliers.loc[suppliers['id']=='AR-PAC','공급중단시작월']='2039-01'
    suppliers.loc[suppliers['연공급상한_톤'].eq(0),'사용_1']=0
    plants=pd.DataFrame(PLANTS,columns=['국가','건설비_백만USD','건설기간_개월','생산능력_톤년','변동비_USD톤','고정비_백만USD년'])
    plants['원료확보_SC6톤월']=187500/12*3;plants['설비실효율']=1.;plants['원료투입계수']=7.5;plants['운송비_USD톤']=0.;plants['최대공장수']=3;plants['비PFE_1']=1;plants['평가재료가치_USD톤']=15000.;plants['초기가동률']=.30;plants['증산기간_개월']=6
    month=pd.DataFrame({'월':months.strftime('%Y-%m'),'수요_톤':1250.,'신규투자예산_백만USD':[125. if d.month in (1,4,7,10) else 0. for d in months],
        '수입가격배율':1.,'공급가능량배율':1.,'원료확보배율':1.,
        '핵심광물목표':[REG_MIN.get(d.year,.5) for d in months],
        '강화목표':[REG_BAT.get(d.year,.85) for d in months], '사용자목표':.60})
    config={'시작월':'2027-01','종료월':'2035-12','목표월':'2033-07','시나리오':'핵심광물','초기투자잔액_백만USD':0.,'최소보유현금_백만USD':0.,'연할인율':.08,'난수시드':42,'난수사용':False,'조달가격변동폭':.10,'공장건설비변동폭':.15,'운영비변동폭':.10,'수요변동폭':.08,'예산변동폭':.10,'공급배정률_최소':.15,'공급배정률_최대':.35,'건설기간_최소개월':36,'건설기간_최대개월':72,'착공_분기첫달만':True,'환율_KRW_USD':1400.,'생성시드':None,'생성강도':'기준값',
            '계약기간_난수최소개월':24,'계약기간_난수최대개월':72,
            '최소구매율_난수최소':0.05,'최소구매율_난수최대':0.25}
    sm=pd.DataFrame([{'월':mo,'id':r['id'],'기업':r['기업'],'가격배율':1.,'공급배율':1.} for mo in month['월'] for _,r in suppliers.iterrows()])
    pm=pd.DataFrame([{'월':mo,'국가':r['국가'],'원료배율':1.} for mo in month['월'] for _,r in plants.iterrows()])
    return {'config':config,'suppliers':suppliers,'plants':plants,'months':month,'supplier_monthly':sm,'plant_monthly':pm}

def clone(data):
    return {'config':copy.deepcopy(data['config']),**{k:data[k].copy(deep=True) for k in ['suppliers','plants','months','supplier_monthly','plant_monthly']}}

def prepare(data):
    d=clone(data);cfg=d['config'];su=d['suppliers'];pl=d['plants'];mo=d['months']
    for df, cols in [(su,['비PFE_1','단가_USD톤','연공급상한_톤','운송비_USD톤','평가재료가치_USD톤','월변경상한_톤','사용_1','최소계약기간_개월','계약최소구매율']),
                    (pl,['건설비_백만USD','건설기간_개월','생산능력_톤년','변동비_USD톤','고정비_백만USD년','원료확보_SC6톤월','설비실효율','원료투입계수','운송비_USD톤','최대공장수','비PFE_1','평가재료가치_USD톤','초기가동률','증산기간_개월']),
                    (mo,['수요_톤','신규투자예산_백만USD','수입가격배율','공급가능량배율','원료확보배율','핵심광물목표','강화목표','사용자목표'])]:
        for col in cols: df[col]=pd.to_numeric(df[col],errors='raise');
        if df[cols].isna().any().any():raise ValueError('입력에 빈 숫자 셀이 있습니다.')
    if (su[['단가_USD톤','연공급상한_톤','운송비_USD톤','월변경상한_톤']]<0).any().any() or (pl[['건설비_백만USD','생산능력_톤년','변동비_USD톤','고정비_백만USD년','원료확보_SC6톤월','운송비_USD톤']]<0).any().any() or (mo[['수요_톤','신규투자예산_백만USD']]<0).any().any():raise ValueError('음수 수요·가격·예산·원가 불가')
    if (su['최소계약기간_개월']<1).any() or (su['최소계약기간_개월']>360).any() or (su['최소계약기간_개월']%1!=0).any(): raise ValueError('최소 계약기간은 1~360개월의 정수여야 합니다')
    if not su['계약최소구매율'].between(0,1).all(): raise ValueError('계약 최소구매율은 0~1 사이여야 합니다')
    if not su['비PFE_1'].isin([0,1]).all() or not pl['비PFE_1'].isin([0,1]).all(): raise ValueError('PFE 적격값 0 또는 1')
    if not ((mo[['핵심광물목표','강화목표','사용자목표']]>=0)&(mo[['핵심광물목표','강화목표','사용자목표']]<=1)).all().all():raise ValueError('규제 비율은 0~1')
    if not pl['설비실효율'].between(0.01,1.0).all():raise ValueError('설비 실효율은 0.01~1이어야 합니다')
    if (pl['원료투입계수']<=0).any() or (pl['건설기간_개월']<1).any() or (pl['최대공장수']<0).any():raise ValueError('공장 기간·원료계수 입력 확인')
    if su['id'].duplicated().any() or pl['국가'].duplicated().any():raise ValueError('기업 ID·공장국 중복')
    for field in ('공급시작월','공급중단시작월'):
        if field not in su.columns:su[field]='2027-01' if field=='공급시작월' else ''
        su[field]=su[field].fillna('').astype(str).str.strip()
        if not su[field].map(lambda v: v=='' or bool(__import__('re').fullmatch(r'\d{4}-(0[1-9]|1[0-2])',v))).all():
            raise ValueError(f'{field}: YYYY-MM 형식으로 입력하세요')
    for _,r in su.iterrows():
        if r['공급시작월'] and r['공급중단시작월'] and r['공급시작월']>=r['공급중단시작월']:
            raise ValueError(f"{r['기업']}: 공급 시작월은 중단 시작월보다 빨라야 합니다")

    if float(cfg.get('최소보유현금_백만USD',0))<0:raise ValueError('최소 보유현금은 0 이상이어야 합니다')
    if not (0<=float(cfg['연할인율'])<1):raise ValueError('할인율 0~1')
    mo=mo[(mo['월']>=cfg['시작월'])&(mo['월']<=cfg['종료월'])].sort_values('월').reset_index(drop=True)
    if mo.empty or cfg['목표월'] not in set(mo['월']):raise ValueError('목표월은 분석기간 안이어야 합니다')
    if len(mo)>360:raise ValueError('분석기간은 최대 360개월(30년)입니다')
    expected=pd.date_range(mo['월'].iloc[0]+'-01',mo['월'].iloc[-1]+'-01',freq='MS').strftime('%Y-%m').tolist()
    if mo['월'].tolist()!=expected:raise ValueError('월별 입력이 누락 또는 중복되었습니다')
    if bool(cfg['난수사용']):
        rng=np.random.default_rng(int(cfg['난수시드']))
        def change(column,df,pct):df[column]=df[column].astype(float)*rng.uniform(1-float(pct),1+float(pct),len(df))
        change('단가_USD톤',su,cfg['조달가격변동폭']);change('건설비_백만USD',pl,cfg['공장건설비변동폭']);change('변동비_USD톤',pl,cfg['운영비변동폭']);change('고정비_백만USD년',pl,cfg['운영비변동폭']);change('수요_톤',mo,cfg['수요변동폭']);change('신규투자예산_백만USD',mo,cfg['예산변동폭'])
        # Base production and allocation assumptions; unknown suppliers stay zero
        for idx,r in su.iterrows():
            if r['연공급상한_톤']>0:su.loc[idx,'연공급상한_톤']=r['연공급상한_톤']*rng.uniform(float(cfg['공급배정률_최소']),float(cfg['공급배정률_최대']))/.25
        pl['건설기간_개월']=rng.integers(int(cfg['건설기간_최소개월']),int(cfg['건설기간_최대개월'])+1,len(pl))
    for key, columns in [('supplier_monthly',['가격배율','공급배율']),('plant_monthly',['원료배율'])]:
        for col in columns:
            d[key][col]=pd.to_numeric(d[key][col],errors='raise')
            if (d[key][col]<0).any() or d[key][col].isna().any():raise ValueError(f'{key}: 배율은 음수가 아닌 수치만 입력 가능합니다')
    if d['supplier_monthly'].duplicated(['월','id']).any() or d['plant_monthly'].duplicated(['월','국가']).any():raise ValueError('월별 기업/공장 데이터 중복')
    d['months']=mo;return d

def target_vector(d):
    sc=d['config']['시나리오'];m=d['months'];col={'핵심광물':'핵심광물목표','강화':'강화목표','직접입력':'사용자목표'}
    if sc=='고정60':return np.full(len(m),.6)
    if sc not in col:raise ValueError('규제 시나리오를 선택하세요')
    return m[col[sc]].to_numpy(float)

def solve(raw, mode='strict',force_country=None,time_limit=45):
    """strict: deadline enforce. diagnostic: minimize violations with explicit slacks.
    earliest: optimize earliest stable compliance using monotone binary active states.
    flex_budget: minimize total quarterly injections, deciding procurement and investment jointly.
    """
    d=prepare(raw);cfg=d['config'];su=d['suppliers'];pl=d['plants'];m=d['months'];T=len(m);S=len(su);C=len(pl);goal=target_vector(d)
    deadline=m['월'].tolist().index(cfg['목표월']);
    if mode=='earliest':goal=np.maximum(goal,float(goal[deadline]))  # explicit target-month threshold maintained thereafter
    discount=(1+float(cfg['연할인율']))**(-np.arange(T)/12)
    # Decision vars x(S*T), y(C*T), z(C*T), budget(T), optional deficits(T), optional compliance slack(T), optional earliest flags(T)
    x0=0;y0=S*T;z0=y0+C*T;b0=z0+C*T;dem0=b0+T;comp0=dem0+(T if mode=='diagnostic' else 0);flag0=comp0+(T if mode=='diagnostic' else 0)
    budget_var=flag0+(T if mode=='earliest' else 0)
    # Contract decision binary variables: active and new start for each supplier/month.
    contract_active0=budget_var+(T if mode=='flex_budget' else 1 if mode=='min_budget' else 0)
    contract_start0=contract_active0+S*T
    N=contract_start0+S*T
    ix=lambda i,t:x0+i*T+t;iy=lambda c,t:y0+c*T+t;iz=lambda c,t:z0+c*T+t;ib=lambda t:b0+t
    idem=lambda t:dem0+t;icomp=lambda t:comp0+t;iflag=lambda t:flag0+t
    ia=lambda i,t:contract_active0+i*T+t
    istart=lambda i,t:contract_start0+i*T+t
    obj=np.zeros(N);lo=np.zeros(N);hi=np.full(N,np.inf);integer=np.zeros(N,dtype=np.int32)
    lo[b0:b0+T]=float(cfg.get('최소보유현금_백만USD',0.))
    if mode=='flex_budget':
        # Quarterly discretionary cash injections: separate funding variables (MUSD).
        # No funding outside quarter-start months. Never assume unmodeled sales revenue.
        for t, month in enumerate(m['월']):
            k=budget_var+t
            if month[-2:] in ('01','04','07','10'):
                hi[k]=1e5
            else:
                hi[k]=0.0
    if mode=='min_budget':
        # First stage: minimize uniform quarterly all-expenses cash injection, ignoring the provided monthly schedule.
        # This is NOT a minimum quarterly company cash funding.
        obj[:]=0.; obj[budget_var]=1.0; hi[budget_var]=1e5
    month_price=m['수입가격배율'].to_numpy(float);month_supply=m['공급가능량배율'].to_numpy(float);feedmul=m['원료확보배율'].to_numpy(float)
    sm=d['supplier_monthly'].set_index(['월','id']);pm=d['plant_monthly'].set_index(['월','국가'])
    months=m['월'].tolist();supplier_price=np.ones((S,T));supplier_cap=np.ones((S,T));country_feed=np.ones((C,T))
    for i,r in su.iterrows():
        for t,month in enumerate(months):
            if (month,r['id']) in sm.index:
                supplier_price[i,t]=float(sm.loc[(month,r['id']),'가격배율']);supplier_cap[i,t]=float(sm.loc[(month,r['id']),'공급배율'])
    for c,r in pl.iterrows():
        for t,month in enumerate(months):
            if (month,r['국가']) in pm.index:country_feed[c,t]=float(pm.loc[(month,r['국가']),'원료배율'])
    for i,r in su.iterrows():
        for t in range(T):
            k=ix(i,t);available=bool((not r['공급시작월'] or months[t]>=r['공급시작월']) and (not r['공급중단시작월'] or months[t]<r['공급중단시작월']));hi[k]=max(0.,float(r['연공급상한_톤'])/12*month_supply[t]*supplier_cap[i,t])*int(r['사용_1'])*int(available);obj[k]=discount[t]*(float(r['단가_USD톤'])*month_price[t]*supplier_price[i,t]+float(r['운송비_USD톤']))/1e6
            integer[ia(i,t)]=1;integer[istart(i,t)]=1
            # Disallow an active contract when supplier is unavailable.
            hi[ia(i,t)]=1 if hi[k]>1e-8 else 0
            duration=int(r['최소계약기간_개월'])
            # Do not start commitments that expire beyond horizon or supplier's known exit.
            endidx=min(T,next((u for u,mon in enumerate(months) if r['공급중단시작월'] and mon>=r['공급중단시작월']),T))
            hi[istart(i,t)]=int(hi[ia(i,t)]>0 and t+duration<=endidx and t+duration<=T)
    ramp=[]
    for c,r in pl.iterrows():
        lead=int(r['건설기간_개월']);ramp_month=int(r['증산기간_개월']);base=float(r['초기가동률'])
        def factor(age):return 0. if age<0 else (1. if ramp_month<=1 else min(1.,base+(1-base)*age/(ramp_month-1)))
        weights=np.array([[factor(t-j-lead) for j in range(T)] for t in range(T)])
        ramp.append(weights)
        for t in range(T):
            obj[iy(c,t)]=discount[t]*(float(r['변동비_USD톤'])+float(r['운송비_USD톤']))/1e6
            k=iz(c,t);integer[k]=1;hi[k]=float(r['최대공장수']) if (not cfg['착공_분기첫달만'] or int(m.loc[t,'월'][5:7]) in (1,4,7,10)) and (t+lead<T) else 0.
            obj[k]=discount[t]*float(r['건설비_백만USD'])+sum(discount[v]*float(r['고정비_백만USD년'])/12 for v in range(t+lead,T))
    if mode=='min_budget':
        obj[:]=0.; obj[budget_var]=1.0
    elif mode=='flex_budget':
        # Minimize total nominal external funding across the horizon.
        # Infinitesimal early-funding penalty breaks equal-cost timing ties.
        # This is a joint MILP over cash injections, suppliers and plants.
        obj[:]=0.
        for t in range(T):
            if hi[budget_var+t]>0: obj[budget_var+t]=1.0+1e-7*(T-t)/T
    if mode=='diagnostic':
        # Lexicographically prioritize meeting demand, then compliance, then cost (millions).
        for t in range(T):
            obj[idem(t)]=200000*discount[t];obj[icomp(t)]=10000*discount[t] if t>=deadline else 0.;hi[icomp(t)]=np.inf if t>=deadline else 0.
    if mode=='earliest':
        # Big penalty ensures earliest feasible *sustained* compliance wins over spending.
        # Starting from deadline is NOT required here; minimum first meaningful target per target month.
        for t in range(T):
            k=iflag(t);integer[k]=1;hi[k]=1 if t>=0 else 0;obj[k]=-100000*(T-t)/T
    rows=[];cols=[];values=[];lower=[];upper=[]
    def add(terms,low=-np.inf,high=np.inf):
        rid=len(lower)
        for k,v in terms.items():
            if abs(v)>1e-12:rows.append(rid);cols.append(k);values.append(v)
        lower.append(low);upper.append(high)
    for t in range(T):
        demand=float(m.loc[t,'수요_톤']);terms={**{ix(i,t):1. for i in range(S)},**{iy(c,t):1. for c in range(C)}}
        if mode=='diagnostic':terms[idem(t)]=1.
        add(terms,demand,demand)
        # Unified enterprise cash account: purchases + variable production + operating
        # fixed costs of every commissioned plant + groundbreaking CAPEX.
        # Costs are undiscounted nominal USD millions, exactly matching monthly results.
        terms={ib(t):1.}
        for i in range(S):
            unit_price=float(su.loc[i,'단가_USD톤'])*month_price[t]*supplier_price[i,t]+float(su.loc[i,'운송비_USD톤'])
            terms[ix(i,t)]=unit_price/1e6
        for c in range(C):
            r=pl.iloc[c]
            terms[iy(c,t)]=(float(r['변동비_USD톤'])+float(r['운송비_USD톤']))/1e6
            terms[iz(c,t)]=float(r['건설비_백만USD'])
            # A new unit incurs its monthly fixed OPEX from the month it first operates.
            lead=int(r['건설기간_개월'])
            if t>=lead:
                for start in range(t-lead+1):
                    terms[iz(c,start)]=terms.get(iz(c,start),0.)+float(r['고정비_백만USD년'])/12
        if t>0:terms[ib(t-1)]=-1.
        if mode=='min_budget':
            if int(m.loc[t,'월'][5:7]) in (1,4,7,10):terms[budget_var]=-1.
            rhs=float(cfg['초기투자잔액_백만USD']) if t==0 else 0.
        elif mode=='flex_budget':
            terms[budget_var+t]=-1.
            rhs=float(cfg['초기투자잔액_백만USD']) if t==0 else 0.
        else:
            rhs=float(m.loc[t,'신규투자예산_백만USD'])+(float(cfg['초기투자잔액_백만USD']) if t==0 else 0.)
        add(terms,rhs,rhs)
        # fixed valuation ratio; separate from operating costs, *research proxy only*
        eligible_terms={ix(i,t):(float(su.loc[i,'비PFE_1'])-goal[t])*float(su.loc[i,'평가재료가치_USD톤'])/15000 for i in range(S)}
        eligible_terms.update({iy(c,t):(float(pl.loc[c,'비PFE_1'])-goal[t])*float(pl.loc[c,'평가재료가치_USD톤'])/15000 for c in range(C)})
        if mode in ('strict','min_budget','flex_budget') and t>=deadline: add(eligible_terms,0,np.inf)
        if mode=='diagnostic' and t>=deadline:
            eligible_terms[icomp(t)]=1.;add(eligible_terms,0,np.inf)
        if mode=='earliest':
            # If flag is zero, relax compliance by the max possible possible demand-valued magnitude.
            relax=max(1.,demand*max(1.,float(su['평가재료가치_USD톤'].max())/15000,float(pl['평가재료가치_USD톤'].max())/15000))
            eligible_terms[iflag(t)]=-relax
            add(eligible_terms,low=-relax)
            if t<T-1:add({iflag(t):1,iflag(t+1):-1},high=0)
    # Contract schedule: activate only by signing, honor minimum term and minimum take-or-pay quantity.
    # A new signing must be an actual off->on transition, and may recur after an expired contract.
    for i,r in su.iterrows():
        duration=int(r['최소계약기간_개월']);minratio=float(r['계약최소구매율'])
        for t in range(T):
            a=ia(i,t); start=istart(i,t); x=ix(i,t)
            currentcap=float(hi[x])
            # Active contract if and only if this month has a valid positive supply and enough minimum offtake.
            add({x:1.,a:-currentcap},high=0.)
            # Minimum purchase is based on current available capacity to accommodate outages.
            if minratio>0: add({x:1.,a:-currentcap*minratio},low=0.)
            if t==0:
                add({a:1.,start:-1.},low=0.,high=0.)
            else:
                prev=ia(i,t-1)
                add({a:1.,prev:-1.,start:-1.},high=0.)  # active transition requires signature
                add({start:1.,a:-1.},high=0.)
                add({start:1.,prev:1.},high=1.)
            # Each signature forces active procurement throughout the agreed minimum term.
            if hi[start]>0:
                for u in range(t,t+duration):
                    add({ia(i,u):1.,start:-1.},low=0.)
    for i,r in su.iterrows():
        change_limit=float(r['월변경상한_톤'])
        if change_limit>0:
            for t in range(1,T):
                if hi[ix(i,t)]>0 and hi[ix(i,t-1)]>0:
                    big=max(float(hi[ix(i,t)]),float(hi[ix(i,t-1)]))
                    # Change limit binds only when the supplier contract is active in BOTH months.
                    add({ix(i,t):1,ix(i,t-1):-1,ia(i,t):big,ia(i,t-1):big},high=change_limit+2*big)
                    add({ix(i,t-1):1,ix(i,t):-1,ia(i,t):big,ia(i,t-1):big},high=change_limit+2*big)
    for c,r in pl.iterrows():
        lead=int(r['건설기간_개월']);capacity=float(r['생산능력_톤년'])*float(r['설비실효율'])/12
        for t in range(T):
            terms={iy(c,t):1.}
            for start in range(t+1):
                if ramp[c][t,start]>0:terms[iz(c,start)]=-capacity*ramp[c][t,start]
            add(terms,high=0)
            add({iy(c,t):float(r['원료투입계수'])},high=float(r['원료확보_SC6톤월'])*feedmul[t]*country_feed[c,t])
        add({iz(c,t):1 for t in range(T)},high=int(r['최대공장수']))
        if force_country is not None and r['국가']==force_country: add({iz(c,t):1 for t in range(T)},low=1)
    if force_country and force_country not in list(pl['국가']):raise ValueError('알 수 없는 국가')
    A=coo_matrix((values,(rows,cols)),shape=(len(lower),N)).tocsr()
    res=milp(obj,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(A,lower,upper),options={'time_limit':time_limit,'mip_rel_gap':0.002})
    output={'required_quarter_budget_MUSD':float(res.x[budget_var]) if mode=='min_budget' and res.x is not None else None,'status':int(res.status),'message':str(res.message),'feasible':res.x is not None,'optimal':res.status==0,'mode':mode,'data':d,'deadline':deadline,'goal':goal,'raw_objective':None if res.fun is None else float(res.fun)}
    if res.x is None:return output
    v=res.x;ship=[];prod=[];factories=[];monthly=[];budget=[];contracts=[];disc_cost=0.
    for i,r in su.iterrows():
        for t,month in enumerate(months):
            contracts.append({'t':t,'월':month,'기업':r['기업'],'id':r['id'],'계약유지_1':int(round(v[ia(i,t)])),'신규계약_1':int(round(v[istart(i,t)])),'최소계약기간_개월':int(r['최소계약기간_개월']),'계약최소구매율':float(r['계약최소구매율'])})
    for c,r in pl.iterrows():
        for t in range(T):
            n=int(round(v[iz(c,t)]))
            if n:
                for unit in range(n):factories.append({'국가':r['국가'],'공장':f"{r['국가']} 정제공장 {unit+1}호",'착공_idx':t,'가동_idx':t+int(r['건설기간_개월']),'착공월':m.loc[t,'월'],'가동월':m.loc[t+int(r['건설기간_개월']),'월'],'건설비_MUSD':float(r['건설비_백만USD'])})
    for t in range(T):
        imp=0.;opex=0.;own=0.;mat=0.;good=0.;volgood=0.
        for i,r in su.iterrows():
            qty=max(0.,float(v[ix(i,t)]));p=float(r['단가_USD톤'])*month_price[t]*supplier_price[i,t]+float(r['운송비_USD톤']);imp+=qty*p/1e6
            value=qty*float(r['평가재료가치_USD톤']);mat+=value;good+=value*int(r['비PFE_1']);volgood+=qty*int(r['비PFE_1'])
            ship.append({'t':t,'월':m.loc[t,'월'],'기업':r['기업'],'국가':r['국가'],'비PFE_1':int(r['비PFE_1']),'수입량_톤':qty,'수입비_MUSD':qty*p/1e6,'단가_USD톤':p})
        for c,r in pl.iterrows():
            qty=max(0.,float(v[iy(c,t)]));num=sum(int(round(v[iz(c,s)])) for s in range(t+1) if t-s>=int(r['건설기간_개월']))
            cost=qty*(float(r['변동비_USD톤'])+float(r['운송비_USD톤']))/1e6+num*float(r['고정비_백만USD년'])/12
            opex+=cost;own+=qty;value=qty*float(r['평가재료가치_USD톤']);mat+=value;good+=value*int(r['비PFE_1']);volgood+=qty*int(r['비PFE_1'])
            prod.append({'t':t,'월':m.loc[t,'월'],'국가':r['국가'],'기업':r['국가']+' 신규공장','비PFE_1':int(r['비PFE_1']),'생산량_톤':qty,'가동공장수':num,'운영비_MUSD':cost})
        capex=sum(float(pl.loc[c,'건설비_백만USD'])*int(round(v[iz(c,t)])) for c in range(C))
        share=good/mat if mat>0 else 0.;require=float(goal[t]);need=t>=deadline
        monthly.append({'t':t,'월':m.loc[t,'월'],'수요_톤':float(m.loc[t,'수요_톤']),'미공급_톤':max(0,float(v[idem(t)])) if mode=='diagnostic' else 0.,'자체생산_톤':own,'비PFE비율':share,'법률참고목표':require,'적용목표':require if need else 0.,'목표적용':need,'목표충족':share+1e-6>=require if need else True,'비PFE톤':volgood,'수입비_MUSD':imp,'운영비_MUSD':opex,'건설비_MUSD':capex,'총지출_MUSD':imp+opex+capex,'준수부족_평가톤':max(0.,float(v[icomp(t)])) if mode=='diagnostic' and need else 0.})
        incoming=(float(v[budget_var+t]) if mode=='flex_budget' else float(v[budget_var]) if mode=='min_budget' and int(m.loc[t,'월'][5:7]) in (1,4,7,10) else float(m.loc[t,'신규투자예산_백만USD']) if mode not in ('min_budget','flex_budget') else 0.)
        budget.append({'t':t,'월':m.loc[t,'월'],'신규예산_MUSD':incoming,'수입비지출_MUSD':imp,'운영비지출_MUSD':opex,'건설비지출_MUSD':capex,'총지출_MUSD':imp+opex+capex,'월말현금잔액_MUSD':max(0,float(v[ib(t)])),'월말투자잔액_MUSD':max(0,float(v[ib(t)]))})
        disc_cost+=discount[t]*(imp+opex+capex)
    output.update({'contracts':pd.DataFrame(contracts),'shipments':pd.DataFrame(ship),'production':pd.DataFrame(prod),'factories':pd.DataFrame(factories,columns=['국가','공장','착공_idx','가동_idx','착공월','가동월','건설비_MUSD']),'summary':pd.DataFrame(monthly),'budgets':pd.DataFrame(budget),'total_cost_MUSD':disc_cost,'total_cash_cost_MUSD':pd.DataFrame(monthly)['총지출_MUSD'].sum(),'total_plants':len(factories),'total_funding_MUSD':float(pd.DataFrame(budget)['신규예산_MUSD'].sum()),'ending_cash_MUSD':float(v[ib(T-1)]),'flags':np.array([round(v[iflag(t)]) for t in range(T)]) if mode=='earliest' else None})
    return output

def optimize_flexible_funding(data,time_limit=80):
    """Jointly minimize nominal external quarterly funding while meeting demand,
    regulation, contracts, plant limits, and month-end minimum cash.
    A feasible result without proven optimality is never presented as a minimum.
    """
    result=solve(data,mode='flex_budget',time_limit=time_limit)
    result['interpretation']='분기별 배정금액을 직접 선택하여 전 기간 명목 외부 자금조달 합계를 최소화한 공동 MILP'
    return result

def run_analysis(data,time_limit=45):
    base=solve(data,'strict',time_limit=time_limit)
    if base['feasible']:
        base['interpretation']='목표 시점부터 제약을 준수하는 계획';return base
    fallback=solve(data,'diagnostic',time_limit=time_limit)
    fallback['strict_failure']=base['message'];fallback['interpretation']='원래 목표는 실행 불가능. 미달성 최소화 보조 최적화 결과 (규제 적격 계획 아님)'
    return fallback

def earliest_stable(data,time_limit=45):
    return solve(data,mode='earliest',time_limit=time_limit)

def funds_to_maintain(result):
    """Report *planned* recurring spend, NOT hypothetical secured financing or exact min external cash."""
    s=result['summary'];f=result['factories'];post=s[s['목표적용']]
    if post.empty:return None
    return {'평균월조달운영비_MUSD':float((post['수입비_MUSD']+post['운영비_MUSD']).mean()),'최대월조달운영비_MUSD':float((post['수입비_MUSD']+post['운영비_MUSD']).max()),'해당기간추가건설비_MUSD':float(post['건설비_MUSD'].sum())}


def optimize_required_budget(data,time_limit=60):
    """Optimize minimum UNIFORM quarterly ALL-EXPENSES cash funding, then min discounted total cost at that funding.
    If first-stage solver stops without optimality proof, result is explicitly approximate.
    """
    first=solve(data,mode='min_budget',time_limit=time_limit)
    if not first['feasible']:
        first['interpretation']='지정 기한까지의 규제 충족은 예산상한을 풀어도 확인되지 않았습니다. 공급량·건설기간·규제를 점검하세요.'
        return first
    q=max(0.,float(first['required_quarter_budget_MUSD']))
    altered=clone(data); months=altered['months']; monthnums=months['월'].str[-2:]
    months['신규투자예산_백만USD']=np.where(monthnums.isin(['01','04','07','10']),q+0.02,0.)
    second=solve(altered,mode='strict',time_limit=time_limit)
    if not second['feasible']:
        first['interpretation']='최소 투자예산 후보는 계산되었지만, 해당 예산에서 총비용 최소화 재계산을 완료하지 못했습니다.'
        return first
    second['required_quarter_budget_MUSD']=q
    second['budget_first_stage_status']=first['status']
    second['budget_first_stage_message']=first['message']
    second['budget_proven_minimum']=first['optimal']
    second['interpretation']='기한 충족을 위한 균등 분기 통합 운영예산 최소화 후 총비용 최소화'
    return second
