"""Synchronized Plotly replay: one time axis, actual MILP output only."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG='#0B1728';PANEL='#12243A';FG='#EFF6FF';MUT='#9FB5CF';GRID='#283D55';PFE='#FB7185';NON='#38BDF8';OWN='#FBBF64';TEAL='#41DDC2'
COORD={'중국':(35.9,104.2),'칠레':(-33.45,-70.66),'호주':(-25.27,133.78),'미국':(39.83,-98.58),'한국':(36.48,127.91),'캐나다':(56.13,-106.35),'아르헨티나':(-38.4,-63.6),'브라질':(-14.2,-51.9),'인도네시아':(-0.78,113.9),'캐나다 신규 정제시설':(56.13,-106.35)}

def events(r,t):
    months=r['summary']; month=months.iloc[t]['월']; f=r['factories'];es=[]
    for _,p in f.iterrows():
        if p['착공_idx']==t:es.append(f"🏗️ {p['국가']} 정제공장 착공 · 설비투자 결정")
        if p['가동_idx']==t:es.append(f"⚡ {p['국가']} 정제공장 가동 시작 · 월별 자체 생산량에 반영")
    if t==r['deadline']:es.append('🎯 설정한 규제 달성 목표월 도래')
    for _,supplier in r['data']['suppliers'].iterrows():
        if str(supplier.get('공급중단시작월',''))==month and int(supplier.get('사용_1',1)) and float(supplier['연공급상한_톤'])>0:
            es.append(f"⛔ {supplier['기업']} 계약 공급 종료 · 대체 조달계획 전환")
    if months.iloc[t]['비PFE비율']+1e-6>=months.iloc[t]['적용목표'] and months.iloc[t]['목표적용'] and (t==0 or not months.iloc[t-1]['목표충족']):es.append('✅ 현재 조달 비PFE 기준 충족')
    if month.endswith(('-01','-04','-07','-10')):es.append(f"💰 분기 통합 현금예산 배정 · 수입·생산·건설비 차감")
    return es

def status_text(r,t):
    f=r['factories'];ans=[]
    for _,p in f.iterrows():
        if t<p['착공_idx']:continue
        if t<p['가동_idx']:
            progress=round((t-p['착공_idx'])/max(1,p['가동_idx']-p['착공_idx'])*100)
            ans.append(f"{p['국가']} 공장 건설 중 · 경과기간 {progress}% · {p['가동월']} 가동 예정")
        else:ans.append(f"{p['국가']} 공장 가동 중 · {p['가동월']}부터 생산")
    return ans or ['신규 정제공장 없음 · 기존 기업 조달']

def replay(r,exchange_rate=1400.):
    s=r['summary'];b=r['budgets'];xs=r['shipments'];ys=r['production'];T=len(s)
    suppliers=r['data']['suppliers'];plants=r['data']['plants'];entities=suppliers['기업'].tolist()+[v+' 자체생산' for v in plants['국가']]
    types=['PFE 기업' if int(row['비PFE_1'])==0 else '비PFE 기업' for _,row in suppliers.iterrows()]+['자체 공장']*len(plants)
    colors=[PFE if k=='PFE 기업' else NON if k=='비PFE 기업' else OWN for k in types]
    ship=xs.pivot_table(index='t',columns='기업',values='수입량_톤',aggfunc='sum').reindex(index=range(T),columns=suppliers['기업'],fill_value=0).fillna(0)
    prod=ys.pivot_table(index='t',columns='국가',values='생산량_톤',aggfunc='sum').reindex(index=range(T),columns=plants['국가'],fill_value=0).fillna(0)
    qty=np.concatenate([ship.values,prod.values],axis=1)
    fig=make_subplots(rows=2,cols=2,column_widths=[.60,.40],row_heights=[.61,.39],vertical_spacing=.20,horizontal_spacing=.13,
                      specs=[[{'type':'geo'},{'type':'xy'}],[{'type':'xy'},{'type':'xy'}]],subplot_titles=['한국 중심 글로벌 조달 경로','공급기업별 월 조달량','비PFE 비율 · 규제 목표','통합 현금 유입·총비용·잔액'])
    trace_indices=[]
    source=[]
    for country in list(dict.fromkeys(suppliers['국가'].tolist()+plants['국가'].tolist())):
        for kind,color in [('PFE 기업',PFE),('비PFE 기업',NON),('자체 공장',OWN)]:
            lat,lon=COORD.get(country,COORD.get(str(country).split(' ')[0],COORD['한국']));klat,klon=COORD['한국']
            source.append((country,kind))
            fig.add_trace(go.Scattergeo(lat=[lat,klat],lon=[lon,klon],mode='lines+markers',line=dict(color=color,width=2),marker=dict(size=[7,1],color=color),opacity=0,showlegend=False,hovertemplate=f'{country} → 한국<br>{kind}<extra></extra>'),row=1,col=1)
            trace_indices.append(len(fig.data)-1)
    fig.add_trace(go.Scattergeo(lat=[COORD['한국'][0]],lon=[COORD['한국'][1]],mode='markers+text',text=['대한민국'],textposition='bottom center',marker=dict(size=13,color=TEAL),textfont=dict(color=FG),showlegend=False),row=1,col=1)
    bars=[]
    for j,(name,col) in enumerate(zip(entities,colors)):
        fig.add_trace(go.Bar(y=[name],x=[0.],orientation='h',marker=dict(color=col),showlegend=False,hovertemplate='%{y}: %{x:,.0f}톤<extra></extra>'),row=1,col=2)
        bars.append(len(fig.data)-1)
    years=[f"{v[:4]}" if v.endswith('-01') else '' for v in s['월']]
    fig.add_trace(go.Scatter(x=list(range(T)),y=s['비PFE비율']*100,mode='lines',name='조달 적격비율',line=dict(color=TEAL,width=3)),row=2,col=1)
    fig.add_trace(go.Scatter(x=list(range(T)),y=s['법률참고목표']*100,mode='lines',name='규제 참고선',line=dict(color=OWN,width=2,dash='dash'),line_shape='hv'),row=2,col=1)
    fig.add_trace(go.Scatter(x=[0],y=[s.iloc[0]['비PFE비율']*100],mode='markers',marker=dict(color=FG,size=11,line=dict(color=TEAL,width=2)),showlegend=False),row=2,col=1);regcursor=len(fig.data)-1
    fig.add_trace(go.Bar(x=list(range(T)),y=b['신규예산_MUSD']*exchange_rate/100,name='예산 유입',marker=dict(color='#2DB8A5',opacity=.25),showlegend=True),row=2,col=2)
    fig.add_trace(go.Bar(x=list(range(T)),y=-b['총지출_MUSD']*exchange_rate/100,name='월 지출',marker=dict(color='#FF9D6C',opacity=.30),showlegend=True),row=2,col=2)
    fig.add_trace(go.Scatter(x=list(range(T)),y=b['월말현금잔액_MUSD']*exchange_rate/100,mode='lines',name='월말 현금잔액',line=dict(color=NON,width=3)),row=2,col=2)
    fig.add_trace(go.Scatter(x=[0],y=[b.iloc[0]['월말현금잔액_MUSD']*exchange_rate/100],mode='markers',marker=dict(color=FG,size=11),showlegend=False),row=2,col=2);cashcursor=len(fig.data)-1
    xmaximum=max(1300,qty.max()*1.2)
    fig.update_xaxes(row=1,col=2,range=[0,xmaximum],title_text='톤 / 월',gridcolor=GRID,tickfont=dict(size=10,color=FG))
    fig.update_yaxes(row=1,col=2,automargin=True,tickfont=dict(size=10,color=FG))
    fig.update_xaxes(row=2,col=1,range=[0,T-1],tickmode='array',tickvals=list(range(0,T,12)),ticktext=[s.iloc[i]['월'][:4] for i in range(0,T,12)],gridcolor=GRID)
    fig.update_yaxes(row=2,col=1,range=[0,105],ticksuffix='%',gridcolor=GRID)
    fig.update_xaxes(row=2,col=2,range=[0,T-1],tickmode='array',tickvals=list(range(0,T,12)),ticktext=[s.iloc[i]['월'][:4] for i in range(0,T,12)],gridcolor=GRID)
    fig.update_yaxes(row=2,col=2,title_text='억원 (통합 현금잔액)',gridcolor=GRID)
    fig.update_geos(projection_type='natural earth',showland=True,landcolor='#263B53',showocean=True,oceancolor='#0B1728',showcountries=True,countrycolor='#4C647F',showcoastlines=False,bgcolor=PANEL,projection_scale=.87)
    def country_qty(t,c,k):
        if k=='자체 공장':return float(ys.loc[(ys.t==t)&(ys['국가']==c),'생산량_톤'].sum())
        return float(xs.loc[(xs.t==t)&(xs['국가']==c)&(xs['비PFE_1']==int(k=='비PFE 기업')),'수입량_톤'].sum())
    subplot_labels=list(fig.layout.annotations)
    for anno in subplot_labels: anno.font=dict(size=13,color=FG)
    annotations=[]
    for t in range(T):
        updates=[]
        for country,k in source:
            v=country_qty(t,country,k)
            updates.append(go.Scattergeo(opacity=0.82 if v>.01 else 0,line=dict(width=1.5+5*min(1,v/max(1,s.iloc[t]['수요_톤']))),hovertext=f'{country} · {k}: {v:,.0f}톤'))
        for j in range(len(entities)):updates.append(go.Bar(x=[max(0.,qty[t,j])]))
        updates.append(go.Scatter(x=[t],y=[s.iloc[t]['비PFE비율']*100]))
        updates.append(go.Scatter(x=[t],y=[b.iloc[t]['월말현금잔액_MUSD']*exchange_rate/100]))
        texts=status_text(r,t);ev=events(r,t)
        # Preserve Plotly's four subplot labels. A separate compact header avoids overlap.
        ann=subplot_labels
        annotations.append(ann)
        if t==0:
            for i,up in zip(trace_indices,updates[:len(source)]):fig.data[i].opacity=up.opacity;fig.data[i].line.width=up.line.width
            for i,up in zip(bars,updates[len(source):-2]):fig.data[i].x=up.x
        fig.frames+= (go.Frame(name=str(t),data=updates,traces=trace_indices+bars+[regcursor,cashcursor],layout=go.Layout(annotations=ann)),)
    fig.update_layout(title='',height=1020,margin=dict(l=45,r=38,t=150,b=165),paper_bgcolor=BG,plot_bgcolor=PANEL,font=dict(color=FG,family='Noto Sans KR, Arial'),barmode='overlay',
        legend=dict(orientation='h',y=-.28,x=.01,font=dict(size=10)),annotations=annotations[0],
        updatemenus=[dict(type='buttons',direction='left',showactive=False,x=0.03,y=1.11,xanchor='left',yanchor='bottom',buttons=[
            dict(label='▶ 재생',method='animate',args=[None,dict(frame=dict(duration=260,redraw=True),transition=dict(duration=100),fromcurrent=True,mode='immediate')]),
            dict(label='Ⅱ 일시정지',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])])],
        sliders=[dict(active=0,x=.10,y=-.12,len=.85,currentvalue=dict(prefix='월: ',font=dict(color=TEAL,size=11),visible=True),steps=[dict(label=s.iloc[t]['월'],method='animate',args=[[str(t)],dict(frame=dict(duration=0,redraw=True),transition=dict(duration=0),mode='immediate')]) for t in range(T)])])
    return fig

def plant_gantt(r):
    f=go.Figure();df=r['factories'];T=len(r['summary'])
    if df.empty:
        f.add_annotation(text='이 시나리오에서는 신규 공장 투자가 선택되지 않았습니다.',showarrow=False,font=dict(color=FG,size=17))
    for idx,p in df.iterrows():
        f.add_trace(go.Bar(y=[p['공장']],x=[p['가동_idx']-p['착공_idx']],base=[p['착공_idx']],orientation='h',marker=dict(color='#F0A85B'),name='건설 중',showlegend=idx==0,hovertemplate=f"{p['착공월']} 착공 → {p['가동월']} 가동<extra></extra>"))
        f.add_trace(go.Bar(y=[p['공장']],x=[T-p['가동_idx']],base=[p['가동_idx']],orientation='h',marker=dict(color=TEAL),name='가동 중',showlegend=idx==0))
        f.add_trace(go.Scatter(x=[p['가동_idx']],y=[p['공장']],mode='markers',marker=dict(symbol='star',color=FG,size=12),showlegend=False))
    f.update_layout(barmode='overlay',height=max(240,130+len(df)*85),paper_bgcolor=BG,plot_bgcolor=PANEL,font=dict(color=FG),margin=dict(l=15,r=25,t=36,b=45),legend=dict(orientation='h',y=1.16))
    f.update_xaxes(title='월별 기간',tickmode='array',tickvals=list(range(0,T,12)),ticktext=[r['summary'].iloc[i]['월'][:4] for i in range(0,T,12)],gridcolor=GRID,range=[0,T])
    f.update_yaxes(automargin=True)
    return f

def cost_breakdown(r,exchange_rate=1400.):
    s=r['summary'];fig=go.Figure()
    for k,label,col in [('수입비_MUSD','기존기업 수입비',NON),('운영비_MUSD','신규 공장 운영비',TEAL),('건설비_MUSD','신규 공장 건설비',OWN)]:
        fig.add_trace(go.Bar(x=s['월'],y=s[k]*exchange_rate/100,name=label,marker_color=col))
    fig.update_layout(barmode='stack',height=345,paper_bgcolor=BG,plot_bgcolor=PANEL,font=dict(color=FG),margin=dict(l=15,r=20,t=25,b=40),legend=dict(orientation='h',y=1.16))
    fig.update_xaxes(nticks=10,gridcolor=GRID);fig.update_yaxes(title='월 비용 (억원)',gridcolor=GRID)
    return fig
