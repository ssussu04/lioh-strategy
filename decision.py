"""Auditable decision evidence and procurement/factory Gantt for monthly LiOH optimization."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from visuals import BG,PANEL,FG,MUT,GRID,PFE,NON,OWN,TEAL


def why_month(result,t):
    """Evidence about selected suppliers and plants, not fabricated solver causality."""
    s=result['summary'].iloc[t]
    b=result['budgets'].iloc[t]
    mo=str(s['월'])
    d=result['data']; suppliers=d['suppliers']; m=d['months']
    rows=[]
    imp=result['shipments'].loc[result['shipments']['t']==t]
    for _,v in imp.iterrows():
        if v['수입량_톤']<=.01:continue
        r=suppliers.loc[suppliers['기업']==v['기업']].iloc[0]
        maxcap=float(r['연공급상한_톤'])/12*float(m.iloc[t]['공급가능량배율'])
        sm=d['supplier_monthly']; sm=sm.loc[(sm['월']==mo)&(sm['id']==r['id'])]
        if not sm.empty:maxcap*=float(sm.iloc[0]['공급배율'])
        cap_ratio=100*float(v['수입량_톤'])/maxcap if maxcap>0 else 0.
        eligible='비PFE 충족 분류' if int(v['비PFE_1']) else '비PFE 미충족 분류'
        evidence=(f"월 공급상한 {maxcap:,.0f}톤 · "
                  f"상한 사용률 {cap_ratio:.0f}% · {eligible}")
        rows.append({'월':mo,'선택':str(v['기업']),'결정':'기존 기업 조달','물량_톤':round(float(v['수입량_톤']),1),'계산 근거':evidence})
    for _,p in result['factories'].iterrows():
        if int(p['착공_idx'])==t:
            available_prior=float(b['월말현금잔액_MUSD'])+float(b['총지출_MUSD'])
            rows.append({'월':mo,'선택':str(p['공장']),'결정':'착공','물량_톤':0.,'계산 근거':f"건설비 ${p['건설비_MUSD']:,.0f}M · 해당 월 지출 전 이용 가능 현금 약 ${available_prior:,.0f}M · {p['가동월']} 가동 예정. 국가 간 최적 선택 근거는 별도 재최적화로 검증"})
        if int(p['가동_idx'])==t:
            rows.append({'월':mo,'선택':str(p['공장']),'결정':'가동 개시','물량_톤':0.,'계산 근거':'가동까지 건설기간을 충족했으며 월별 생산상한과 원료 제약을 적용함'})
    if s['목표적용']:
        rows.append({'월':mo,'선택':'PFE 목표','결정':'규제 평가','물량_톤':round(float(s['비PFE톤']),1),'계산 근거':f"비PFE 비용평가 비율 {s['비PFE비율']*100:.1f}% / 목표 {s['법률참고목표']*100:.1f}% (법적 적격 판정 아님)"})
    if not rows:
        rows.append({'월':mo,'선택':'수요 충족','결정':'조달 없음','물량_톤':0.,'계산 근거':'이번 달 추가 의사결정 이벤트가 없습니다.'})
    return pd.DataFrame(rows)


def decision_gantt(result):
    """Monthly Gantt including procurement-active spans and plant lifecycle; each bar has grounded hover evidence."""
    sumdf=result['summary']; T=len(sumdf); months=sumdf['월'].tolist()
    ship=result['shipments']; fct=result['factories']
    fig=go.Figure()
    entities=[]
    colors={0:PFE,1:NON}
    # Show contiguous active spans. Bar base/x are month indices for legible long range plotting.
    for name,part in ship.groupby('기업',sort=False):
        values=part.set_index('t')['수입량_톤'].reindex(range(T),fill_value=0.).to_numpy(float)
        if values.max()<.01:continue
        flag=values>.01
        changes=np.diff(np.r_[False,flag,False].astype(int)); starts=np.where(changes==1)[0]; ends=np.where(changes==-1)[0]
        pfe=int(part.iloc[0]['비PFE_1']); country=str(part.iloc[0]['국가']);y=f'기업 · {name}'
        entities.append(y)
        for start,end in zip(starts,ends):
            avg=values[start:end].mean(); volume=values[start:end].sum()
            color=NON if pfe else PFE
            hover=(f'{name} · {country}<br>{months[start]} ~ {months[end-1]}<br>'
                   f'기간 합계 {volume:,.0f}톤 · 월평균 {avg:,.0f}톤<br>'
                   f'비PFE 분류: {"충족" if pfe else "미충족"}')
            fig.add_trace(go.Bar(y=[y],x=[end-start],base=[start],orientation='h',marker=dict(color=color),
                       width=.66,hovertemplate=hover+'<extra></extra>',showlegend=False))
    # A contract is a separate business commitment from a shipment, so show its true duration.
    contracts=result.get('contracts',pd.DataFrame())
    if not contracts.empty:
        for name,part in contracts.groupby('기업',sort=False):
            values=part.set_index('t')['계약유지_1'].reindex(range(T),fill_value=0).to_numpy(int)
            if values.max()==0:continue
            intervals=np.diff(np.r_[0,values,0]);starts=np.where(intervals==1)[0];ends=np.where(intervals==-1)[0]
            y=f'계약 · {name}';entities.append(y)
            for start,end in zip(starts,ends):
                mo_min=int(part.iloc[0]['최소계약기간_개월']);pct=float(part.iloc[0]['계약최소구매율'])*100
                fig.add_trace(go.Bar(y=[y],x=[end-start],base=[start],orientation='h',width=.48,
                    marker=dict(color='#9A85E8',opacity=.90,line=dict(color='#D5CAFF',width=1)),showlegend=False,
                    hovertemplate=f'{name}<br>계약 유지 {months[start]} ~ {months[end-1]}<br>계약 최소기간 {mo_min}개월 · 월 최소구매율 {pct:.1f}%<extra></extra>'))
    for _,p in fct.iterrows():
        y='공장 · '+str(p['공장']);entities.append(y)
        start=int(p['착공_idx']); act=int(p['가동_idx']); act=min(T,act)
        fig.add_trace(go.Bar(y=[y],x=[act-start],base=[start],orientation='h',width=.7,
            marker=dict(color=OWN),hovertemplate=f"{p['공장']}<br>{p['착공월']} 착공 · 건설비 ${p['건설비_MUSD']:,.0f}M<br>{p['가동월']} 가동 시작<extra></extra>",showlegend=False))
        if act<T:
            fig.add_trace(go.Bar(y=[y],x=[T-act],base=[act],orientation='h',width=.7,marker=dict(color=TEAL),
                hovertemplate=f"{p['공장']} · {p['가동월']} 이후 가동<br>실제 월별 생산량은 아래 상세표 참조<extra></extra>",showlegend=False))
            fig.add_trace(go.Scatter(x=[act],y=[y],mode='markers',marker=dict(size=11,symbol='star',color='#FFFFFF'),showlegend=False,
                                     hovertemplate=f'{p["가동월"]} 공장 가동 개시<extra></extra>'))
        fig.add_trace(go.Scatter(x=[start],y=[y],mode='markers',marker=dict(size=10,symbol='diamond',color='#FFFFFF'),showlegend=False,
                                 hovertemplate=f'{p["착공월"]} 공장 착공<extra></extra>'))
    deadline=result['deadline']
    fig.add_vline(x=deadline,line=dict(color='#FFFFFF',width=1.5,dash='dash'))
    fig.add_annotation(x=deadline,y=1.075,xref='x',yref='paper',showarrow=False,text='목표 시점',font=dict(color='#EAF4FF',size=12),bgcolor='#264158',borderpad=5)
    fig.update_layout(title=dict(text='월별 조달 선택과 공장 투자 일정',font=dict(size=18,color=FG),x=.01,y=.99),
        barmode='overlay',height=max(350,155+len(entities)*36),paper_bgcolor=BG,plot_bgcolor=PANEL,
        font=dict(color=FG,size=12),margin=dict(l=15,r=30,t=95,b=80),showlegend=False,
        annotations=[*fig.layout.annotations])
    fig.update_xaxes(range=[0,T],tickmode='array',tickvals=list(range(0,T,12)),ticktext=[months[i][:4] for i in range(0,T,12)],
                     gridcolor=GRID,title='시간 (월별 의사결정)')
    fig.update_yaxes(categoryorder='array',categoryarray=list(reversed(entities)),automargin=True,tickfont=dict(size=11))
    return fig


def decision_events(result):
    """Event table summarizing genuine MILP decisions. Supplier increases are observed deltas."""
    s=result['summary']; ship=result['shipments']; f=result['factories']; T=len(s)
    records=[]
    for _,p in f.iterrows():
        records.append({'t':int(p['착공_idx']),'월':p['착공월'],'분류':'공장 착공','대상':p['공장'],
            '실제 계산 결과':f"CAPEX ${p['건설비_MUSD']:,.0f}M 집행, {p['가동월']} 가동 예정",
            '설명':'착공 당시 사용 가능한 통합 현금예산 내에서 건설기간·생산능력·규제 목표를 고려한 결정'})
        if int(p['가동_idx'])<T:
            records.append({'t':int(p['가동_idx']),'월':p['가동월'],'분류':'공장 가동','대상':p['공장'],
                '실제 계산 결과':'신규 공장 생산량 조달계획에 반영','설명':'건설기간이 경과해 월별 생산능력·원료 제약 범위에서 생산 가능'})
    pivot=ship.pivot_table(index='t',columns='기업',values='수입량_톤',aggfunc='sum').reindex(range(T),fill_value=0.).fillna(0.)
    for t in range(1,T):
        change=(pivot.iloc[t]-pivot.iloc[t-1])
        for name,delta in change.items():
            if abs(delta)>=max(100.,float(s.iloc[t]['수요_톤'])*.15):
                records.append({'t':t,'월':s.iloc[t]['월'],'분류':'조달 물량 변경','대상':name,
                    '실제 계산 결과':f"전월 대비 {delta:+,.0f}톤 (당월 {pivot.loc[t,name]:,.0f}톤)",
                    '설명':'해당 월의 기업별 공급상한·구매비용·규제 조건을 충족한 최적화 결과. 인과적 단일 원인은 대안 재최적화가 필요'})
    contracts=result.get('contracts',pd.DataFrame())
    if not contracts.empty:
        for _,row in contracts.loc[contracts['신규계약_1'].eq(1)].iterrows():
            records.append({'t':int(row['t']),'월':row['월'],'분류':'공급 계약 체결','대상':row['기업'],
                '실제 계산 결과':f"최소 {int(row['최소계약기간_개월'])}개월 유지, 월 공급가능량의 {100*float(row['계약최소구매율']):.1f}% 이상 구매",
                '설명':'단기 계약 반복 없이 계약기간과 최소 구매량 제약을 포함해 선택된 공급사'})
    for _,firm in result['data']['suppliers'].iterrows():
        end=str(firm.get('공급중단시작월',''))
        if end in set(s['월']):
            records.append({'t':int(s.index[s['월'].eq(end)][0]),'월':end,'분류':'계약 공급 종료','대상':firm['기업'],
                '실제 계산 결과':'해당 월부터 공급상한 0톤',
                '설명':'계약 종료 시점 이후 수입량은 MILP에서 0톤으로 제한되며 대체 공급·공장생산과 함께 최적화'})
    records.append({'t':int(result['deadline']),'월':s.iloc[result['deadline']]['월'],'분류':'목표 평가','대상':'규제 충족',
         '실제 계산 결과':f"비PFE {s.iloc[result['deadline']]['비PFE비율']*100:.1f}% / 기준 {s.iloc[result['deadline']]['법률참고목표']*100:.1f}%",
         '설명':'규제 목표월의 비용평가 비PFE 비율. 법적 적격 판단 아님'})
    return pd.DataFrame(records).sort_values(['t','분류']).drop(columns='t').reset_index(drop=True)
