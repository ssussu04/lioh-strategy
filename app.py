"""LiOH STRATEGY V5 | Korean supply-chain decision control tower."""
import io, json, zipfile, time, copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from model import inputs_default,clone,run_analysis,earliest_stable,solve,funds_to_maintain,optimize_required_budget,optimize_flexible_funding,REG_MIN,REG_BAT
from visuals import replay,plant_gantt,cost_breakdown,events,status_text
from decision import decision_gantt,why_month,decision_events
from scenarios import generate, RANGE_INFO

st.set_page_config(page_title='LiOH STRATEGY | 공급망 전환 관제센터',page_icon='🌐',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html,body,.stApp,[data-testid="stMarkdownContainer"]{font-family:'Noto Sans KR',sans-serif!important}
.stApp{background:radial-gradient(ellipse at 80% 0%,#15374a55,transparent 45%),#091421;color:#F3F7FD}
[data-testid="stSidebar"]{background:#111F31;border-right:1px solid #293F54}
[data-testid="stSidebar"] *{color:#DFEAF9!important}
[data-testid="stAppViewContainer"] .block-container{max-width:1660px;padding-top:1.8rem}
h1,h2,h3{color:#F6FAFF!important;letter-spacing:-.035em}
h1{font-size:2.1rem!important;font-weight:800!important}
h2{font-size:1.6rem!important}
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stMarkdownContainer"] p{color:#D8E5F4}
.stApp [data-testid="stCaptionContainer"] p{color:#A9BDD3!important}
.stApp [data-testid="stMetric"]{min-height:118px;background:linear-gradient(140deg,#192D45,#122235);border:1px solid #33506A;border-radius:16px;padding:18px}
.stApp [data-testid="stMetricLabel"] p{color:#B6CEE6!important;font-weight:600}
.stApp [data-testid="stMetricValue"]{color:#55DED2!important;font-weight:800;font-size:1.85rem}
.stApp .stButton button[kind="primary"]{border:0;background:linear-gradient(90deg,#1767A6,#098F96);color:#FFFFFF;font-weight:800;min-height:3rem;border-radius:12px}
.stApp [data-testid="stDataFrame"]{border:1px solid #294860;border-radius:11px}
.stApp [data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
[data-testid="stExpander"]{background:#12243B55;border:1px solid #34506C;border-radius:12px}
.hero{padding:27px 28px;border-radius:18px;border:1px solid #355773;background:linear-gradient(110deg,#17354B,#102033);margin-bottom:20px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:2px;color:#50DBC9}
.headline{font-size:29px;font-weight:800;color:#F8FCFF;margin:7px 0}
.subtitle{font-size:13px;color:#B9CEE4}
.notice{padding:16px;border-radius:12px;background:#132C3E;border:1px solid #2A6875;color:#DCEBF4;margin:12px 0}
.event{background:#1B3245;padding:11px 15px;border-left:3px solid #FBBF64;border-radius:8px;margin:6px 0;color:#EDF7FF}
.section-title{color:#F4FAFF;font-size:19px;font-weight:800;padding:24px 0 12px;margin-top:16px;border-top:1px solid #273E53}
/* Result presentation: deliberate breathing room, not a wall of cards */
.stApp [data-testid="stHorizontalBlock"]{gap:1.15rem!important}
.stApp [data-testid="stMetric"]{margin-bottom:.4rem}
.stApp [data-testid="stMetricLabel"]{padding-bottom:8px}
.stApp [data-testid="stMetricValue"]{line-height:1.28}
.stApp [data-testid="stPlotlyChart"]{margin-top:12px;margin-bottom:24px}
.stApp [data-testid="stAlert"]{margin-block:12px}
.stApp [data-testid="stTabs"]{margin-top:18px}
[data-testid="stTabs"] button p{font-weight:700;color:#BCD3E9!important}
[data-testid="stTabs"] button[aria-selected="true"] p{color:#61E2D0!important}
</style>''',unsafe_allow_html=True)

if 'analysis_mode' not in st.session_state:st.session_state.analysis_mode='목표시점 입력 → 최소 필요 통합예산 계산'
if 'budget_ui' not in st.session_state:st.session_state.budget_ui=125.0
if 'target_ui' not in st.session_state:st.session_state.target_ui=50.0
if 'inputs' not in st.session_state:st.session_state.inputs=inputs_default()
if 'result' not in st.session_state:st.session_state.result=None
if 'earliest' not in st.session_state:st.session_state.earliest=None
if 'last_exec' not in st.session_state:st.session_state.last_exec=''
if 'input_fingerprint' not in st.session_state:st.session_state.input_fingerprint=''
if 'version' not in st.session_state:st.session_state.version=0
if 'replay_cache' not in st.session_state:st.session_state.replay_cache=None
if 'funding_options' not in st.session_state:st.session_state.funding_options={}
if 'funding_style' not in st.session_state:st.session_state.funding_style='두 방식 비교'
D=st.session_state.inputs;cfg=D['config']
if '환율_KRW_USD' not in cfg:cfg['환율_KRW_USD']=1400.
def krw(amount_musd):
    return float(amount_musd)*float(cfg['환율_KRW_USD'])/100. # USD million -> 100 million KRW
def won_label(amount_musd):
    val=krw(amount_musd)
    return f'{val/10000:,.3f}조 원' if abs(val)>=10000 else f'{val:,.1f}억 원'


def fingerprint(data):
    import hashlib
    b=json.dumps({'config':data['config'],**{k:data[k].to_json(orient='records') for k in ['suppliers','plants','months','supplier_monthly','plant_monthly']}},ensure_ascii=False,sort_keys=True,default=str)
    return hashlib.sha256(b.encode()).hexdigest()

def funding_chart(funding_options, rate):
    """Quarterly time series comparing external injection and month-end balances in KRW 100M."""
    f=go.Figure()
    palette=[('#48C5F4','#1577B7'),('#56E3CC','#179884')]
    for idx,(label,r) in enumerate(funding_options.items()):
        if not r or not r.get('feasible'):continue
        b=r['budgets'].copy()
        b['분기']=pd.PeriodIndex(pd.to_datetime(b['월']+'-01'),freq='Q').astype(str).str.replace('Q',' Q',regex=False)
        grouped=b.groupby('분기',sort=True).agg({'신규예산_MUSD':'sum','총지출_MUSD':'sum','월말현금잔액_MUSD':'last'}).reset_index()
        col=palette[idx%len(palette)]
        f.add_trace(go.Bar(x=grouped['분기'],y=grouped['신규예산_MUSD']*rate/100,name=label+' · 자금배정',marker_color=col[0],opacity=.85,hovertemplate='%{x}<br>%{y:,.1f}억원<extra>'+label+'</extra>'))
        f.add_trace(go.Scatter(x=grouped['분기'],y=grouped['월말현금잔액_MUSD']*rate/100,name=label+' · 기말잔액',yaxis='y2',mode='lines',line={'color':col[1],'width':2.8},hovertemplate='%{x}<br>%{y:,.1f}억원<extra>'+label+'</extra>'))
    f.update_layout(height=395,barmode='group',paper_bgcolor='#0F1F32',plot_bgcolor='#0F1F32',font={'color':'#E8F1FA','family':'Noto Sans KR, sans-serif'},margin={'l':55,'r':65,'t':20,'b':85},legend={'orientation':'h','y':-0.33,'x':0},xaxis={'title':None,'tickangle':-40,'nticks':14,'gridcolor':'#203449'},yaxis={'title':'분기 자금배정 (억원)','gridcolor':'#2B4054','zerolinecolor':'#516477'},yaxis2={'title':'분기말 현금잔액 (억원)','overlaying':'y','side':'right','showgrid':False},hovermode='x unified')
    return f


def funding_summary(funding_options):
    records=[]
    for label,r in funding_options.items():
        if not r or not r.get('feasible'):continue
        b=r['budgets']; first=int(b['t'].min())
        records.append({'방식':label,'총 외부 배정예산_억원':round(krw(b['신규예산_MUSD'].sum()),2),'최대 분기 배정액_억원':round(krw(b.groupby((b['월'].str[:4].astype(int)*4 + (b['월'].str[5:7].astype(int)-1)//3))['신규예산_MUSD'].sum().max()),2),'기말 현금잔액_억원':round(krw(b.iloc[-1]['월말현금잔액_MUSD']),2),'할인 총비용_억원':round(krw(r['total_cost_MUSD']),2),'최적성 증명':('완료' if r['optimal'] and (r['mode']!='strict' or r.get('budget_proven_minimum',False)) else '시간제한·근사')})
    return pd.DataFrame(records)


def excel_result_bytes(r):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        for name,key in [('월별현황','summary'),('계약일정','contracts'),('기업별수입','shipments'),('공장생산','production'),('공장건설','factories'),('예산이월','budgets')]:r[key].to_excel(w,sheet_name=name,index=False)
    return buf.getvalue()

def excel_inputs_bytes(data):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        pd.DataFrame([{'항목':k,'값':v} for k,v in data['config'].items()]).to_excel(w,sheet_name='설정',index=False)
        for name,df in [('공급기업',data['suppliers']),('공장',data['plants']),('월별입력',data['months']),('기업월별',data['supplier_monthly']),('공장월별',data['plant_monthly'])]:df.to_excel(w,sheet_name=name,index=False)
    return buf.getvalue()

def output_zip(r):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('최적화_결과.xlsx',excel_result_bytes(r))
        z.writestr('공급망_통합리플레이.html',(st.session_state.replay_cache or replay(r,exchange_rate=float(cfg['환율_KRW_USD']))).to_html(include_plotlyjs=True,full_html=True))
        z.writestr('기업선택_공장건설_통합간트.html',decision_gantt(r).to_html(include_plotlyjs=True,full_html=True))
        z.writestr('월별_의사결정일지.csv',decision_events(r).to_csv(index=False).encode('utf-8-sig'))
        z.writestr('비용_분해.html',cost_breakdown(r,exchange_rate=float(cfg['환율_KRW_USD'])).to_html(include_plotlyjs=True,full_html=True))
    return out.getvalue()

def choose_preset(kind):
    new=inputs_default()
    if kind=='성장 시나리오':
        n=len(new['months']);new['months']['수요_톤']=np.linspace(1250,3750,n)
    elif kind=='원료 제약 시나리오':
        new['plants'].loc[new['plants']['국가']=='칠레','원료확보_SC6톤월']=2200.
    elif kind=='투자예산 제약 시나리오':
        new['months'].loc[new['months']['월'].str[-2:].isin(['01','04','07','10']),'신규투자예산_백만USD']=30.
    elif kind=='강화 규제 시나리오':new['config']['시나리오']='강화'
    st.session_state.inputs=new;st.session_state.result=None;st.session_state.earliest=None;st.session_state.replay_cache=None;st.session_state.version+=1

with st.sidebar:
    st.markdown('## ◈ LiOH STRATEGY')
    st.caption('공급망 의사결정 플랫폼')
    cfg['환율_KRW_USD']=st.number_input('표시 환율 (원 / USD)',min_value=1.,max_value=10000.,value=float(cfg['환율_KRW_USD']),step=10.,help='원본 입력과 MILP는 USD를 유지하고 화면만 원화로 환산합니다. 동일 환율의 일괄 환산은 최적해를 변경하지 않습니다.')
    st.divider()
    page=st.radio('메뉴',['📊 종합 현황','▶ 공급망 전환 리플레이','🏢 우리 기업 · 월별 수요/예산','🚢 공급기업 관리','🏭 해외 정제공장 투자','⚖️ PFE 규제·달성 목표','🧪 난수·시나리오 설정','🛠 모든 입력값 편집'],label_visibility='collapsed')
    st.divider()
    with st.expander('데이터 관리 정책',expanded=False):
        st.caption('추가 공급사와 시나리오 수치는 입력 가정이며 검증된 실거래 계약정보가 아닙니다. PFE 구분은 비용기반 공급망 관리지표로, 미국 법령 적격성 판정과 별개입니다.')
    with st.expander('💾 시나리오 저장·불러오기'):
        st.caption('기업·공장·수요·예산·규제 입력조건을 저장하고 복원합니다. 최적화 결과는 별도 다운로드를 사용하세요.')
        snapshot={'config':D['config'],**{k:D[k].to_dict(orient='records') for k in ['suppliers','plants','months','supplier_monthly','plant_monthly']}}
        st.download_button('현재 시나리오 저장 (JSON)',json.dumps(snapshot,ensure_ascii=False,indent=2,default=str).encode('utf-8'),'LiOH_시나리오.json',mime='application/json')
        upload=st.file_uploader('저장한 시나리오 불러오기 (JSON)',type=['json'],key='upload_project')
        if upload is not None and st.button('불러오기 적용'):
            try:
                restored=json.loads(upload.getvalue().decode('utf-8'))
                if not all(k in restored for k in ['config','suppliers','plants','months']):raise ValueError('필수 입력 테이블이 없는 파일입니다')
                st.session_state.inputs={'config':restored['config'],**{k:pd.DataFrame(restored[k]) for k in ['suppliers','plants','months','supplier_monthly','plant_monthly']}}
                st.session_state.result=None;st.session_state.earliest=None;st.session_state.replay_cache=None;st.session_state.version+=1;st.rerun()
            except Exception as ex:st.error(f'입력파일 읽기 오류: {ex}')

st.markdown('<div class="hero"><div class="eyebrow">LITHIUM HYDROXIDE · DECISION INTELLIGENCE</div><div class="headline">수산화리튬 공급망 전환 관제센터</div><div class="subtitle">월별 수입·자체 생산 · 분기별 통합 예산과 이월 · 국가별 공장 착공·완공 · 목표 달성 전망</div></div>',unsafe_allow_html=True)

if page=='🏢 우리 기업 · 월별 수요/예산':
    st.header('우리 기업 · 수요 및 통합 예산')
    st.caption('수요·조달비·공장 운영비·건설비를 모두 충당할 자금과 월별 조달환경을 설정하세요.')
    a,b,c=st.columns(3)
    a.metric('분석기간',f"{cfg['시작월']} ~ {cfg['종료월']}")
    b.metric('목표 달성 월',cfg['목표월'])
    c.metric('초기 공장 수','0개')
    initial_won=st.number_input('초기 회사 현금잔액 (억원)',min_value=0.,value=round(krw(cfg['초기투자잔액_백만USD']),2),step=10.)
    cfg['초기투자잔액_백만USD']=initial_won*100/float(cfg['환율_KRW_USD'])
    reserve_won=st.number_input('최소 보유현금 (억원)',min_value=0.,max_value=1000000.,value=round(krw(cfg.get('최소보유현금_백만USD',0.)),2),step=10.,help='모든 월말에 반드시 남겨야 하는 최소 현금잔액입니다. 수입비, 공장 운영비, 건설비 지출 이후에도 유지됩니다.')
    cfg['최소보유현금_백만USD']=reserve_won*100/float(cfg['환율_KRW_USD'])
    col1,col2=st.columns(2)
    with col1:
        monthly_d=st.number_input('모든 월에 적용할 수요량 (톤)',min_value=0.,value=1250.,step=50.)
        if st.button('월 수요 일괄 적용'):
            D['months']['수요_톤']=monthly_d;st.rerun()
    with col2:
        new_won=st.number_input('분기마다 배정할 회사 통합 예산 (억원)',min_value=0.,value=round(krw(125.),2),step=10.)
        new_b=new_won*100/float(cfg['환율_KRW_USD'])
        if st.button('매 분기 첫 달에 예산 일괄 편성'):
            D['months']['신규투자예산_백만USD']=[new_b if x[-2:] in ['01','04','07','10'] else 0 for x in D['months']['월']];st.rerun()
    st.markdown('#### 월별 수요·통합 예산 직접 편집')
    st.caption('월별 예산 칸은 원한다면 분기 중 다른 달에도 자금을 배정할 수 있습니다. 배정되지 않은 금액은 자동으로 이월됩니다.')
    cols=['월','수요_톤','신규투자예산_백만USD','수입가격배율','공급가능량배율','원료확보배율']
    edited=st.data_editor(D['months'][cols],hide_index=True,use_container_width=True,disabled=['월'],num_rows='fixed',key='monthly_input_'+str(st.session_state.version))
    for k in cols[1:]:D['months'][k]=edited[k].to_numpy()
    st.download_button('현재 기업·월별 입력표 Excel 다운로드',excel_inputs_bytes(D),'LiOH_V5_입력값.xlsx')

elif page=='🚢 공급기업 관리':
    st.header('공급기업별 조달 조건')
    st.caption('최소계약기간_개월(기본 48)과 계약최소구매율(기본 0.15)을 직접 수정할 수 있습니다. 계약 시작 후 기간 동안 월별 최소 구매가 적용됩니다.')
    countries=st.multiselect('국가 필터',sorted(D['suppliers']['국가'].unique().tolist()),default=sorted(D['suppliers']['국가'].unique().tolist()))
    st.dataframe(D['suppliers'][D['suppliers']['국가'].isin(countries)][['국가','기업','비PFE_1','단가_USD톤','연공급상한_톤','최소계약기간_개월','계약최소구매율','공급시작월','공급중단시작월']],hide_index=True,use_container_width=True)
    
    st.markdown('#### 전체 공급기업 입력값 수정')
    D['suppliers']=st.data_editor(D['suppliers'],hide_index=True,num_rows='dynamic',use_container_width=True,key='sup_editor_'+str(st.session_state.version))
    with st.expander('데이터 구분 · 계약정보',expanded=False):
        st.caption('기존 공급사 데이터와 신규 시뮬레이션 계약 입력은 출처_주의 열로 식별됩니다. 이 화면의 공급량·가격은 수정 가능한 계획 입력값입니다.')
        st.dataframe(D['suppliers'][['기업','출처_주의']],hide_index=True,use_container_width=True)
    st.caption('공급중단시작월은 그 달부터 수입 0톤입니다. 계약기간은 최소 유지 개월 수, 계약최소구매율 0.15는 그 달 공급가능량의 15% 이상 구매한다는 뜻입니다. 계약 만료 전의 조기 중단은 허용하지 않습니다.')
    with st.expander('공급기업 추가 · 계약 시작/종료',expanded=False):
        st.caption('신규 업체를 추가하면 계약 일정과 월별 공급배율 입력도 자동 생성합니다.')
        with st.form('add_supplier_form'):
            n1,n2=st.columns(2)
            new_name=n1.text_input('기업명',value='신규 공급 파트너')
            new_country=n2.text_input('공급국',value='호주')
            n3,n4=st.columns(2)
            new_price=n3.number_input('톤당 구매단가 (USD)',min_value=0.,value=16000.)
            new_cap=n4.number_input('연간 공급한도 (톤)',min_value=0.,value=5000.)
            n5,n6=st.columns(2)
            new_start=n5.text_input('공급 개시월',value='2027-01')
            new_stop=n6.text_input('공급 중단 시작월 (없으면 공란)',value='')
            new_eligible=st.selectbox('비PFE 적격 설정',[0,1],index=1)
            ca,cb=st.columns(2)
            new_term=ca.number_input('최소 계약 유지기간 (개월)',min_value=1,max_value=360,value=48,step=1)
            new_minbuy=cb.number_input('계약 중 월별 최소 구매율 (%)',min_value=0.,max_value=100.,value=15.,step=1.)
            submitted=st.form_submit_button('공급기업 추가',type='primary')
            if submitted:
                import uuid
                sid='NEW-'+uuid.uuid4().hex[:8].upper()
                r={col:None for col in D['suppliers'].columns}
                r.update({'id':sid,'기업':new_name.strip(),'국가':new_country.strip(),'비PFE_1':new_eligible,'단가_USD톤':new_price,'연공급상한_톤':new_cap,'출처_주의':'scenario_assumption','운송비_USD톤':0.,'평가재료가치_USD톤':15000.,'월변경상한_톤':0.,'사용_1':1,'공급시작월':new_start,'공급중단시작월':new_stop,'최소계약기간_개월':int(new_term),'계약최소구매율':new_minbuy/100.})
                D['suppliers']=pd.concat([D['suppliers'],pd.DataFrame([r])],ignore_index=True)
                sm=pd.DataFrame([{'월':mo,'id':sid,'기업':new_name.strip(),'가격배율':1.,'공급배율':1.} for mo in D['months']['월']])
                D['supplier_monthly']=pd.concat([D['supplier_monthly'],sm],ignore_index=True)
                st.session_state.version+=1;st.rerun()
    with st.expander('기업별 월별 공급·가격 변동 편집'):
        sid=st.selectbox('수정할 기업',D['suppliers']['id'].tolist(),format_func=lambda v: D['suppliers'].set_index('id').loc[v,'기업'])
        mask=D['supplier_monthly']['id'].eq(sid)
        block=D['supplier_monthly'].loc[mask,['월','가격배율','공급배율']].copy()
        revised=st.data_editor(block,hide_index=True,disabled=['월'],use_container_width=True,key='supplier_monthly_'+str(st.session_state.version)+'_'+sid)
        D['supplier_monthly'].loc[mask,['가격배율','공급배율']]=revised[['가격배율','공급배율']].to_numpy()

elif page=='🏭 해외 정제공장 투자':
    st.header('해외 신규 정제공장 투자조건')
    st.caption('투자 후보별 건설비·생산능력·운영비·건설기간을 설정하세요.')
    a,b,c=st.columns(3)
    for col,r in zip([a,b,c],D['plants'].to_dict('records')):
        with col:
            st.markdown(f"#### {r['국가']} 정제공장")
            st.metric('건설비',won_label(r['건설비_백만USD']))
            st.caption(f"건설기간 {r['건설기간_개월']}개월 · 연 {r['생산능력_톤년']:,.0f}톤")
    st.markdown('#### 국가별 모든 투자 입력값')
    D['plants']=st.data_editor(D['plants'],hide_index=True,num_rows='dynamic',use_container_width=True,key='plant_editor_'+str(st.session_state.version))
    with st.expander('새로운 공장 후보 추가',expanded=False):
        with st.form('add_plant_form'):
            p1,p2=st.columns(2)
            pn=p1.text_input('투자국/후보명',value='캐나다 신규 정제시설')
            pc=p2.number_input('건설비 (백만 USD)',min_value=1.,value=600.)
            p3,p4=st.columns(2)
            ple=p3.number_input('건설기간 (개월)',min_value=1,max_value=120,value=48)
            pcap=p4.number_input('연간 생산능력 (톤)',min_value=1.,value=25000.)
            p5,p6=st.columns(2)
            pop=p5.number_input('톤당 생산 변동비 (USD)',min_value=0.,value=20000.)
            pfix=p6.number_input('연 고정비 (백만 USD)',min_value=0.,value=22.)
            if st.form_submit_button('공장 후보 추가',type='primary'):
                name=pn.strip()
                if not name:st.error('후보명을 입력하세요.')
                elif name in D['plants']['국가'].tolist():st.error('같은 이름의 공장 후보가 존재합니다.')
                else:
                    r={col:None for col in D['plants'].columns}
                    r.update({'국가':name,'건설비_백만USD':pc,'건설기간_개월':ple,'생산능력_톤년':pcap,'변동비_USD톤':pop,'고정비_백만USD년':pfix,'원료확보_SC6톤월':pcap/12*7.5,'설비실효율':.9,'원료투입계수':7.5,'운송비_USD톤':0.,'최대공장수':2,'비PFE_1':1,'평가재료가치_USD톤':15000.,'초기가동률':.3,'증산기간_개월':6})
                    D['plants']=pd.concat([D['plants'],pd.DataFrame([r])],ignore_index=True)
                    pm=pd.DataFrame([{'월':mo,'국가':name,'원료배율':1.} for mo in D['months']['월']])
                    D['plant_monthly']=pd.concat([D['plant_monthly'],pm],ignore_index=True)
                    st.session_state.version+=1;st.rerun()
    st.caption('원료 확보량 및 설비효율을 생산 상한에 반영합니다. 완공 후 초기 증산기간도 적용합니다.')
    cfg['착공_분기첫달만']=st.checkbox('공장 착공은 분기 첫 달만 가능',value=bool(cfg['착공_분기첫달만']))
    with st.expander('국가별 월 원료 확보량 배율 편집'):
        country=st.selectbox('원료 확보량을 수정할 국가',D['plants']['국가'].tolist())
        mask=D['plant_monthly']['국가'].eq(country)
        block=D['plant_monthly'].loc[mask,['월','원료배율']].copy()
        revised=st.data_editor(block,hide_index=True,disabled=['월'],use_container_width=True,key='plant_monthly_'+str(st.session_state.version)+'_'+country)
        D['plant_monthly'].loc[mask,['원료배율']]=revised[['원료배율']].to_numpy()

elif page=='⚖️ PFE 규제·달성 목표':
    st.header('PFE 규제 · 목표 달성 시점')
    st.caption('연도별 비PFE 목표 기준과 목표 달성 시점을 선택하세요.')
    v=['핵심광물','강화','고정60','직접입력'];labels={'핵심광물':'핵심광물 법정 일정 참고','강화':'배터리 부품 기준 차용 · 강화 시나리오','고정60':'60% 고정','직접입력':'사용자 맞춤 목표'}
    cfg['시나리오']=st.selectbox('PFE 조달 목표 기준',v,index=v.index(cfg['시나리오']),format_func=lambda k:labels[k])
    allmonths=D['months']['월'].tolist()
    a,b,c=st.columns(3)
    cfg['시작월']=a.selectbox('분석 시작월',allmonths,index=allmonths.index(cfg['시작월']))
    cfg['종료월']=b.selectbox('분석 종료월',allmonths,index=allmonths.index(cfg['종료월']))
    cfg['목표월']=c.selectbox('규제 목표 달성월',allmonths,index=allmonths.index(cfg['목표월']))
    st.caption('최대 30년(360개월). 2027~2050년을 포함하는 장기 계획을 계산할 수 있습니다.')
    st.caption('시작월 ≤ 목표월 ≤ 종료월. 목표월부터 이후 모든 월의 연도별 기준을 만족하도록 계획을 계산합니다.')
    st.markdown('#### 월별 규제 목표 수정')
    cols=['월','핵심광물목표','강화목표','사용자목표']
    ed=st.data_editor(D['months'][cols],hide_index=True,use_container_width=True,disabled=['월'],key='reg_editor_'+str(st.session_state.version))
    for k in cols[1:]:D['months'][k]=ed[k].to_numpy()

elif page=='🛠 모든 입력값 편집':
    st.header('모든 입력값 · 개별 수정 관리')
    st.info('표의 각 셀을 클릭하면 개별 수정할 수 있습니다. 난수 시나리오 생성 후 이 화면에서 원하는 값만 직접 보정하세요. 수정한 표가 실제 MILP에 입력됩니다.')
    st.caption('결과는 원화로 표시합니다. 환율 변경 시 분석 화면과 비용 환산표가 갱신됩니다.')
    tabs=st.tabs(['우리 회사 · 전 기간','공급기업 전체','기업별 월간 조건','신규 공장','공장별 월간 원료','분석 설정'])
    with tabs[0]:
        st.subheader('우리 회사 월별 수요·통합 현금예산·규제 목표')
        year=st.selectbox('조회/수정할 연도',sorted(D['months']['월'].str[:4].unique()),key='master_company_year')
        mask=D['months']['월'].str.startswith(year)
        vals=D['months'].loc[mask].copy()
        edited=st.data_editor(vals,hide_index=True,disabled=['월'],use_container_width=True,num_rows='fixed',key=f'master_company_{year}_{st.session_state.version}')
        for column in vals.columns:
            if column!='월':D['months'].loc[mask,column]=edited[column].to_numpy()
        st.caption('연도별 월간 수요·예산·가격·원료 조건을 수정할 수 있습니다.')
    with tabs[1]:
        st.subheader('공급기업 개별 특성')
        st.caption('국가·가격·공급상한·계약 시작 및 중단월 등 모든 항목을 수정할 수 있습니다.')
        D['suppliers']=st.data_editor(D['suppliers'],hide_index=True,use_container_width=True,num_rows='fixed',key=f'master_suppliers_{st.session_state.version}')
        sview=D['suppliers'][['기업','단가_USD톤','운송비_USD톤']].copy()
        for price_col in ['단가_USD톤','운송비_USD톤']:
            sview[price_col.replace('USD','만원')]=(sview[price_col]*float(cfg['환율_KRW_USD'])/10000).round(2)
        st.dataframe(sview[['기업','단가_만원톤','운송비_만원톤']],hide_index=True,use_container_width=True)
    with tabs[2]:
        st.subheader('기업별 월간 공급상한·가격배율')
        a,b=st.columns(2)
        sid=a.selectbox('공급기업',D['suppliers']['id'].tolist(),key='master_supplier_id',format_func=lambda v:str(D['suppliers'].set_index('id').loc[v,'기업']))
        year=b.selectbox('연도',sorted(D['supplier_monthly']['월'].str[:4].unique()),key='master_supplier_year')
        mask=D['supplier_monthly']['id'].eq(sid)&D['supplier_monthly']['월'].str.startswith(year)
        vals=D['supplier_monthly'].loc[mask].copy()
        edited=st.data_editor(vals,hide_index=True,disabled=['월','id','기업'],use_container_width=True,num_rows='fixed',key=f'master_sup_month_{sid}_{year}_{st.session_state.version}')
        for column in ['가격배율','공급배율']:D['supplier_monthly'].loc[mask,column]=edited[column].to_numpy()
        st.caption('공급배율 0은 해당 월 수입 중단, 1은 기준 공급상한입니다.')
    with tabs[3]:
        st.subheader('공장별 생산능력·건설비·효율·원료 제약')
        D['plants']=st.data_editor(D['plants'],hide_index=True,use_container_width=True,num_rows='fixed',key=f'master_plants_{st.session_state.version}')
        pview=D['plants'][['국가','건설비_백만USD','변동비_USD톤','고정비_백만USD년']].copy()
        pview['건설비_억원']=pview['건설비_백만USD'].map(krw).round(1)
        pview['변동비_만원톤']=(pview['변동비_USD톤']*float(cfg['환율_KRW_USD'])/10000).round(1)
        pview['연고정비_억원']=pview['고정비_백만USD년'].map(krw).round(1)
        st.dataframe(pview[['국가','건설비_억원','변동비_만원톤','연고정비_억원']],hide_index=True,use_container_width=True)
    with tabs[4]:
        st.subheader('공장별 월간 원료확보 배율')
        a,b=st.columns(2)
        country=a.selectbox('국가',D['plants']['국가'].tolist(),key='master_plant_country')
        year=b.selectbox('연도',sorted(D['plant_monthly']['월'].str[:4].unique()),key='master_plant_year')
        mask=D['plant_monthly']['국가'].eq(country)&D['plant_monthly']['월'].str.startswith(year)
        vals=D['plant_monthly'].loc[mask].copy()
        edited=st.data_editor(vals,hide_index=True,disabled=['월','국가'],use_container_width=True,num_rows='fixed',key=f'master_plant_month_{country}_{year}_{st.session_state.version}')
        D['plant_monthly'].loc[mask,'원료배율']=edited['원료배율'].to_numpy()
    with tabs[5]:
        st.subheader('전체 설정값 직접 수정')
        st.caption('문자·숫자·체크박스를 각각 개별 편집할 수 있습니다. 규제/날짜 설정은 각 전용 메뉴에서도 변경 가능합니다.')
        hidden={'생성시드','생성강도','생성방식','연수요성장률_가정'}
        for key in list(cfg):
            if key in hidden:continue
            value=cfg[key]
            if isinstance(value,(bool,np.bool_)):
                cfg[key]=st.checkbox(key,value=bool(value),key='config_edit_'+key)
            elif isinstance(value,(int,float,np.integer,np.floating)):
                if '환율' in key:continue
                if key in ('난수시드','건설기간_최소개월','건설기간_최대개월'):
                    cfg[key]=int(st.number_input(key,min_value=0,value=int(value),step=1,key='config_edit_'+key))
                else:
                    cfg[key]=st.number_input(key,min_value=0.,value=float(value),format='%.5f',key='config_edit_'+key)
            elif key=='시나리오':
                opts=['핵심광물','강화','고정60','직접입력']
                cfg[key]=st.selectbox(key,opts,index=opts.index(value) if value in opts else 0,key='config_edit_'+key)
            elif key in ('시작월','종료월','목표월'):
                opts=D['months']['월'].tolist()
                cfg[key]=st.selectbox(key,opts,index=opts.index(value) if value in opts else 0,key='config_edit_'+key)
            else:
                cfg[key]=st.text_input(key,value=str(value),key='config_edit_'+key)
    st.divider()
    st.caption('직접 수정 및 난수 생성 데이터는 좌측 사이드바에서 JSON으로 저장·복원할 수 있습니다. 목표·예산 계산 결과는 변경 후 다시 실행해야 반영됩니다.')

elif page=='🧪 난수·시나리오 설정':
    st.header('현실적 불확실성 · 시나리오 실험')
    st.caption('시드에 따라 수요·조달·투자 조건을 생성합니다. 같은 시드는 같은 입력값을 재현합니다.')
    kind=st.selectbox('검토용 시나리오 선택',['기준 시나리오','성장 시나리오','원료 제약 시나리오','투자예산 제약 시나리오','강화 규제 시나리오'])
    if st.button('선택 시나리오 불러오기',type='primary'):
        choose_preset(kind);st.rerun()
    st.divider()
    st.subheader('🎲 버튼 한 번으로 전체 시나리오 생성')
    st.caption('회사·공급기업·공장·월별 공급차질·설비효율·원료조건을 동시에 생성합니다.')
    intensity=st.selectbox('불확실성 강도',['보수적','표준','스트레스'],index=1,key='random_intensity')
    seed_input=int(st.number_input('생성 시드 · 같은 시드면 같은 데이터',min_value=0,max_value=2147483647,value=int(cfg['난수시드']),step=1,key='generation_seed'))
    st.markdown('#### 기업별 장기계약 난수 범위')
    st.caption('기본 24~72개월은 공급계약 사례를 바탕으로 한 시나리오 범위입니다. 각 기업의 실제 계약조건을 뜻하지 않습니다. 공급기업 관리 화면에서 모두 개별 수정할 수 있습니다.')
    ra,rb=st.columns(2)
    cfg['계약기간_난수최소개월']=int(ra.number_input('최소 계약기간 하한 (개월)',min_value=1,max_value=360,value=int(cfg.get('계약기간_난수최소개월',24)),step=1))
    cfg['계약기간_난수최대개월']=int(rb.number_input('최소 계약기간 상한 (개월)',min_value=1,max_value=360,value=int(cfg.get('계약기간_난수최대개월',72)),step=1))
    rc,rd=st.columns(2)
    cfg['최소구매율_난수최소']=float(rc.number_input('계약 최소구매율 하한 (%)',min_value=0.,max_value=100.,value=round(float(cfg.get('최소구매율_난수최소',.05))*100,2),step=1.))/100
    cfg['최소구매율_난수최대']=float(rd.number_input('계약 최소구매율 상한 (%)',min_value=0.,max_value=100.,value=round(float(cfg.get('최소구매율_난수최대',.25))*100,2),step=1.))/100
    if cfg['계약기간_난수최소개월']>cfg['계약기간_난수최대개월'] or cfg['최소구매율_난수최소']>cfg['최소구매율_난수최대']:
        st.error('난수 범위 하한은 상한보다 작거나 같아야 합니다.')
    if st.button('🎲 전체 조건 원클릭 무작위 생성',type='primary',use_container_width=True):
        latest=generate(seed_input,intensity,contract_ranges={k:cfg[k] for k in ('계약기간_난수최소개월','계약기간_난수최대개월','최소구매율_난수최소','최소구매율_난수최대')})
        # Preserve selected study horizon, target and scenario so the button is not a hidden overwrite.
        for config_key in ('시작월','종료월','목표월','시나리오','환율_KRW_USD','연할인율','최소보유현금_백만USD','계약기간_난수최소개월','계약기간_난수최대개월','최소구매율_난수최소','최소구매율_난수최대'):
            latest['config'][config_key]=cfg[config_key]
        st.session_state.inputs=latest;st.session_state.result=None;st.session_state.earliest=None;st.session_state.replay_cache=None;st.session_state.version+=1;st.rerun()
    if cfg.get('생성시드') is not None:st.success(f"현재 생성 데이터 · 시드 {cfg['생성시드']} · 강도 {cfg.get('생성강도')} · 수요성장률 가정 {100*cfg.get('연수요성장률_가정',0):.2f}%/년")
    st.markdown('#### 필요한 영역만 난수 재생성')
    st.caption('전체 재설정과 달리 선택한 영역만 새 난수값으로 바뀝니다. 다른 영역과 직접 수정한 값은 그대로 유지됩니다. 생성 후에도 모든 값은 직접 수정할 수 있습니다.')
    rc1,rc2,rc3=st.columns(3)
    domains=[('우리 회사', ['months']), ('공급기업', ['suppliers','supplier_monthly']), ('신규 공장', ['plants','plant_monthly'])]
    for col,(label,keys) in zip([rc1,rc2,rc3],domains):
        if col.button(f'🎲 {label}만 생성',use_container_width=True):
            fresh=generate(seed_input,intensity,contract_ranges={k:cfg[k] for k in ('계약기간_난수최소개월','계약기간_난수최대개월','최소구매율_난수최소','최소구매율_난수최대')})
            updated=clone(D)
            for table in keys:updated[table]=fresh[table].copy(deep=True)
            updated['config']['생성시드']=seed_input
            updated['config']['생성강도']=intensity
            updated['config']['난수사용']=False
            if label=='우리 회사':
                updated['config']['연수요성장률_가정']=fresh['config'].get('연수요성장률_가정',0.)
            st.session_state.inputs=updated
            st.session_state.result=None;st.session_state.earliest=None;st.session_state.replay_cache=None
            st.session_state.version+=1
            st.rerun()
    with st.expander('시나리오 생성 범위 상세'):
        st.dataframe(pd.DataFrame([{'변수':k,'가정 범위':v} for k,v in RANGE_INFO.items()]),hide_index=True,use_container_width=True)
        st.caption('생성된 값은 상세 입력표에서 수정할 수 있습니다.')
    st.divider()
    cfg['난수사용']=st.toggle('최적화 실행 직전 추가 난수 교란 (원클릭 생성 후에는 끄기 권장)',value=bool(cfg['난수사용']))
    cfg['난수시드']=int(st.number_input('난수 시드',value=int(cfg['난수시드']),step=1))
    for key in ['조달가격변동폭','공장건설비변동폭','운영비변동폭','수요변동폭','예산변동폭']:
        cfg[key]=st.slider(key.replace('변동폭',' 변동폭'),0.,.50,float(cfg[key]),.01)
    cfg['건설기간_최소개월']=int(st.number_input('건설기간 하한 (개월)',min_value=1,max_value=100,value=int(cfg['건설기간_최소개월'])))
    cfg['건설기간_최대개월']=int(st.number_input('건설기간 상한 (개월)',min_value=1,max_value=120,value=int(cfg['건설기간_최대개월'])))
    st.download_button('입력 데이터 Excel 저장',excel_inputs_bytes(D),'LiOH_V5_입력값.xlsx')

else:
    if page=='📊 종합 현황':st.header('의사결정 대시보드')
    else:st.header('공급망 전환 시뮬레이션 · 통합 리플레이')
    st.markdown('### 자금 안정성 기준')
    reserve_value=st.number_input('모든 월말에 유지할 최소 보유현금 (억원)',min_value=0.,max_value=1000000.,value=round(krw(cfg.get('최소보유현금_백만USD',0.)),2),step=10.,key='reserve_dashboard',help='최적화 제약에 직접 반영됩니다. 기준 미달 시 해당 계획은 허용되지 않습니다.')
    cfg['최소보유현금_백만USD']=reserve_value*100/float(cfg['환율_KRW_USD'])
    st.markdown('### 분석 목적 선택')
    analysis_mode=st.radio('두 가지 MILP 분석 중 선택',
        ['목표시점 입력 → 최소 필요 통합예산 계산','통합예산 입력 → 최초 달성 시점 계산'],
        horizontal=True,key='analysis_mode',help='두 모드는 서로 다른 목적함수를 사용합니다.')
    if analysis_mode.startswith('목표시점'):
        st.info('목표연월부터 규제를 유지하는 자금계획을 계산합니다. **균등 분기예산**과 **분기별 변동예산**을 비교할 수 있으며, 수입비·생산비·고정운영비·건설비가 모두 현금계정에서 차감됩니다.')
        funding_style=st.radio('자금조달 방식', ['두 방식 비교','균등 분기예산만','분기별 변동예산만'],horizontal=True,key='funding_style',help='균등: 최소 공통 분기배정액. 변동: 분기별 배정액을 MILP가 따로 정해 총 명목 외부 조달액 최소화.')
        available=D['months'].loc[D['months']['월'].between(cfg['시작월'],cfg['종료월']),'월'].tolist()
        if available:
            if cfg['목표월'] not in available:cfg['목표월']=available[-1]
            cfg['목표월']=st.selectbox('반드시 달성할 목표 연도·월',available,index=available.index(cfg['목표월']))
    else:
        st.info('분기마다 사용할 수 있는 통합 예산을 정하면, 현재 분석 기간 내에서 **규제 목표를 지속적으로 유지할 수 있는 가장 빠른 시작월**을 계산합니다.')
        cc1,cc2=st.columns(2)
        budget_won=cc1.number_input('분기별 회사 통합 예산 (억원)',min_value=0.,max_value=500000.,value=round(krw(st.session_state.budget_ui),2),step=10.,key='budget_krw_ui')
        st.session_state.budget_ui=budget_won*100/float(cfg['환율_KRW_USD'])
        st.session_state.target_ui=cc2.number_input('지속적으로 달성할 비PFE 조달비율 (%)',min_value=0.,max_value=100.,value=float(st.session_state.target_ui),step=5.)
        st.caption('예산은 매 분기 첫 달에 배정되고 남는 금액은 이월됩니다. 수입비·생산변동비·가동 고정비·공장 건설비가 모두 차감됩니다. 매출 유입은 현재 반영되지 않습니다.')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('분석기간',f"{cfg['시작월']} ~ {cfg['종료월']}")
    c2.metric('분석 기준',cfg['목표월'] if analysis_mode.startswith('목표시점') else f'분기 {won_label(st.session_state.budget_ui)}')
    c3.metric('기본 연간 수요',f"{float(D['months'].iloc[0]['수요_톤'])*12:,.0f}톤")
    c4.metric('PFE 조달 목표 기준',{'핵심광물':'핵심광물','강화':'강화','고정60':'고정 60%','직접입력':'직접 입력'}[cfg['시나리오']])
    with st.expander('분석 실행 설정',expanded=st.session_state.result is None):
        a,b,c=st.columns(3)
        cfg['연할인율']=a.number_input('연 할인율 (%)',min_value=0.,max_value=30.,value=round(float(cfg['연할인율'])*100,2),step=.5)/100
        cfg['난수사용']=b.checkbox('난수 모드',value=bool(cfg['난수사용']))
        cfg['난수시드']=int(c.number_input('Seed',value=int(cfg['난수시드']),step=1))
        st.caption('엄격 규제 목표 불가능 시 목표를 조작하지 않고, 별도 미달성 최소화 계획을 제시합니다.')
    if st.button('▶ 선택한 분석 실행',type='primary',use_container_width=True):
        try:
            if not (cfg['시작월']<=cfg['목표월']<=cfg['종료월']):raise ValueError('시작·목표·종료 월 순서를 확인하세요')
            with st.spinner('월별 수입량과 공장 착공, 통합 현금예산을 MILP로 최적화 중...'):
                if analysis_mode.startswith('목표시점'):
                    funding_options={}
                    if st.session_state.funding_style in ('두 방식 비교','균등 분기예산만'):
                        funding_options['균등 분기예산']=optimize_required_budget(D,time_limit=45)
                    if st.session_state.funding_style in ('두 방식 비교','분기별 변동예산만'):
                        funding_options['분기별 변동예산']=optimize_flexible_funding(D,time_limit=65)
                    feasible_options={name:val for name,val in funding_options.items() if val.get('feasible')}
                    st.session_state.funding_options=funding_options
                    if feasible_options:
                        result=feasible_options.get('균등 분기예산') or next(iter(feasible_options.values()))
                    else:
                        result=next(iter(funding_options.values()))
                        diag=run_analysis(D,time_limit=25)
                        if diag['feasible']:result=diag;result['budget_mode_unresolved']=True
                else:
                    st.session_state.funding_options={}
                    scenario=clone(D)
                    scenario['months']['신규투자예산_백만USD']=[float(st.session_state.budget_ui) if x[-2:] in ['01','04','07','10'] else 0. for x in scenario['months']['월']]
                    scenario['months']['사용자목표']=float(st.session_state.target_ui)/100
                    scenario['config']['시나리오']='직접입력'
                    scenario['config']['목표월']=scenario['config']['시작월']
                    result=earliest_stable(scenario,time_limit=60)
                    if result['feasible'] and not np.any(result['flags']>0.5):
                        result['no_sustained_success']=True
            result['display_mode']=analysis_mode
            st.session_state.result=result;st.session_state.earliest=None;st.session_state.replay_cache=None;st.session_state.input_fingerprint=fingerprint(D)+analysis_mode+str(st.session_state.budget_ui)+str(st.session_state.target_ui)+str(st.session_state.funding_style);st.session_state.last_exec=time.strftime('%Y-%m-%d %H:%M:%S')
            if result['feasible']:st.success('계산 완료. 아래 시뮬레이션과 선택 근거를 확인하세요.')
            else:st.error('수요 충족 자체가 불가능하거나 계산이 종료됐습니다. 입력상한을 확인하세요.')
        except Exception as exc:st.error(f'입력 검증 또는 최적화 오류: {exc}')
    if st.session_state.result is None:
        st.markdown('<div class="notice">최적화 버튼을 누르면 세계지도·기업 조달량·공장 투자·비PFE 규제·예산 흐름이 같은 시간축으로 재생됩니다.</div>',unsafe_allow_html=True)
    else:
        r=st.session_state.result
        if analysis_mode.startswith('목표시점') and len([v for v in st.session_state.funding_options.values() if v.get('feasible')])>=2:
            view_choice=st.radio('지도·간트차트에 표시할 자금조달 전략', ['균등 분기예산','분기별 변동예산'],horizontal=True,key='funding_view')
            r=st.session_state.funding_options[view_choice]
        if fingerprint(D)+analysis_mode+str(st.session_state.budget_ui)+str(st.session_state.target_ui)+str(st.session_state.funding_style)!=st.session_state.input_fingerprint:st.warning('입력값이 변경됐습니다. 아래 결과는 이전 실행값입니다. 최적화를 다시 실행하세요.')
        if not r['feasible']:
            st.error('기본 수요 및 예산 제약을 충족할 수 있는 계획을 찾지 못했습니다. 기업 공급량 또는 분석기간을 수정하세요.')
            st.code(r['message'])
        else:
            s=r['summary'];b=r['budgets'];df=r['factories'];deadline=r['deadline'];sgoal=s.iloc[deadline]
            target_ok=(bool(s.loc[deadline:,'목표충족'].all()) if r['mode']!='earliest' else bool(np.any(r['flags']>0.5))) and float(s['미공급_톤'].sum())<.0001
            if analysis_mode.startswith('목표시점') and r.get('required_quarter_budget_MUSD') is not None:
                st.markdown(f'<div class="notice"><b>균등 분기 통합예산 후보: {won_label(r["required_quarter_budget_MUSD"])}</b> · 목표월 {cfg["목표월"]} · 총비용은 별도 계산</div>',unsafe_allow_html=True)
                if not r.get('budget_proven_minimum',False):st.warning('지정 시간 내 최소예산의 수학적 최적성 증명은 완료되지 않았습니다. 현재값은 계산된 가능해 기준입니다.')
            if analysis_mode.startswith('통합예산'):
                flags=r.get('flags')
                marks=np.where(flags>0.5)[0] if flags is not None else []
                if len(marks):
                    st.success(f'설정한 분기 예산 {won_label(st.session_state.budget_ui)}에서 최초 지속 달성 시점: **{s.iloc[int(marks[0])]["월"]}**')
                else:
                    st.warning(f'설정 예산에서 {cfg["종료월"]}까지 지속적으로 {st.session_state.target_ui:.0f}%를 달성하는 시점은 확인되지 않았습니다. 아래 조달 경로는 가능 계획이며 규제 충족 계획이 아닙니다.')
                if not r['optimal']:st.warning('시간제한 내 최적성 증명이 완료되지 않았으므로 표시된 달성월은 확정된 최단시점이 아닐 수 있습니다.')
            if r.get('budget_mode_unresolved'):st.warning('정한 목표를 최소 통합예산으로 달성할 수 있는 해를 확인하지 못해 기존 예산에 대한 보조계획을 보여줍니다.')
            if r['mode']=='diagnostic':st.warning('목표 규제를 완전히 만족하는 계획을 찾지 못했습니다. 아래 결과는 규제 미달을 최소화한 **보조 계획**이며 적격 최적해가 아닙니다.')
            elif r['mode'] in ('strict','flex_budget'):st.success('목표월부터 분석 종료까지 규제 및 수요를 충족하는 계획을 확인했습니다.')
            if analysis_mode.startswith('목표시점'):
                funding_options=st.session_state.funding_options
                valid={name:val for name,val in funding_options.items() if val.get('feasible')}
                st.markdown('<div class="section-title">필요 자금조달 규모 · 핵심 결과</div>',unsafe_allow_html=True)
                st.caption('균등 배정액은 분기당 금액, 변동배정액은 분석기간 전체의 누적 합계입니다. 두 지표의 단위를 구분하세요.')
                st.markdown('<div class="section-title">자금조달 최적화 결과 · 균등 배정 vs 분기별 필요 배정</div>',unsafe_allow_html=True)
                if valid:
                    kpis=st.columns(3)
                    uniform=valid.get('균등 분기예산')
                    flexible=valid.get('분기별 변동예산')
                    kpis[0].metric('균등 분기 배정액 (억원/분기)',won_label(uniform['required_quarter_budget_MUSD']) if uniform and uniform.get('required_quarter_budget_MUSD') is not None else '미계산')
                    kpis[1].metric('변동배정 누적 총액 (억원)',won_label(flexible['total_funding_MUSD']) if flexible else '미계산')
                    kpis[2].metric('변동배정 기말 잔액 (억원)',won_label(flexible['ending_cash_MUSD']) if flexible else '미계산')
                    if len(valid)>1:
                        st.plotly_chart(funding_chart(valid,float(cfg['환율_KRW_USD'])),use_container_width=True,config={'displaylogo':False})
                        st.dataframe(funding_summary(valid),hide_index=True,use_container_width=True)
                        if not flexible['optimal']:st.warning('분기별 변동예산의 최소 총액은 지정시간 내 최적성 증명을 마치지 못했습니다. 표시값은 실행 가능 계획입니다.')
                    elif flexible:
                        st.plotly_chart(funding_chart(valid,float(cfg['환율_KRW_USD'])),use_container_width=True,config={'displaylogo':False})
                        st.dataframe(funding_summary(valid),hide_index=True,use_container_width=True)
                    if flexible:
                        with st.expander('📅 분기별 필요 자금조달 상세표',expanded=False):
                            cash=flexible['budgets'].copy()
                            cash['분기']=pd.PeriodIndex(pd.to_datetime(cash['월']+'-01'),freq='Q').astype(str)
                            qcash=cash.groupby('분기',sort=True).agg({'신규예산_MUSD':'sum','수입비지출_MUSD':'sum','운영비지출_MUSD':'sum','건설비지출_MUSD':'sum','월말현금잔액_MUSD':'last'}).reset_index()
                            for c in qcash.columns[1:]:qcash[c.replace('_MUSD','_억원')]=qcash.pop(c).map(krw)
                            st.dataframe(qcash,hide_index=True,use_container_width=True)
                            st.download_button('분기별 필요 자금조달 Excel 다운로드',data=excel_result_bytes(flexible),file_name='LiOH_V5_변동자금조달_결과.xlsx')
                else:
                    st.warning('요청한 자금조달 모델에서 실행 가능한 계획을 확인하지 못했습니다. 상세 오류를 확인하세요.')
                st.caption('균등 방식은 분기 공통 배정액 최소화 후 할인비용 최소화입니다. 변동 방식은 분기별 배정액·공급기업·공장 투자를 공동 결정하여 전 기간 외부 조달총액(명목)을 최소화합니다. 목적함수가 달라 조달 및 투자계획도 달라질 수 있습니다. 매출·차입이자·자금조달비용은 포함하지 않습니다.')
            a,bcol,cc,dd=st.columns(4)
            a.metric('현재가치 총비용',won_label(r['total_cost_MUSD']))
            bcol.metric('새로 건설한 공장',f"{r['total_plants']}개")
            cc.metric('기준월 비PFE 비율',f"{sgoal['비PFE비율']*100:.1f}%",f"목표 {sgoal['법률참고목표']*100:.0f}%")
            dd.metric('규제 목표 유지', '충족' if target_ok else '미달성')
            st.caption('통합예산은 기업 수입비·자체생산 변동비·가동 중 공장의 고정운영비·공장 건설비를 모두 차감합니다. 현재 모델에 매출 유입은 없으며 분기별 자금배정을 외부 조달 가능 금액으로 가정합니다.')
            reserve_musd=float(r['data']['config'].get('최소보유현금_백만USD',0))
            reserve_gap=b['월말현금잔액_MUSD'].to_numpy(float)-reserve_musd
            min_idx=int(np.argmin(reserve_gap))
            min_gap=float(reserve_gap[min_idx])
            monthly_spend=float(b['총지출_MUSD'].mean())
            month_buffer=min_gap/max(monthly_spend,1e-9)
            st.markdown('<div class="section-title">자금 안정성 · 최소 보유현금 관리</div>',unsafe_allow_html=True)
            cash_cols=st.columns(3)
            cash_cols[0].metric('최소 보유현금 기준',won_label(reserve_musd))
            cash_cols[1].metric('전체 기간 최소 안전여유',won_label(min_gap))
            cash_cols[2].metric('현금잔액 최저 시점',str(b.iloc[min_idx]['월']))
            if min_gap < -1e-3:
                st.error('최소 보유현금 제약 위반이 검출되었습니다. 결과를 제출 전에 재검증하세요.')
            elif month_buffer < 0.5:
                st.warning(f'자금 여유 주의: {b.iloc[min_idx]["월"]}의 현금잔액이 최소 보유현금 기준에 근접합니다. 평균 월 지출 대비 여유 {month_buffer:.2f}개월분.')
            elif month_buffer < 2:
                st.info(f'자금 여유 관찰: 최저 시점 {b.iloc[min_idx]["월"]}, 평균 월 지출 대비 안전여유 {month_buffer:.1f}개월분.')
            else:
                st.success(f'자금 여유 안정: 분석기간의 최소 현금 안전여유 {won_label(min_gap)}. 전 기간 보유현금 기준을 만족합니다.')
            st.caption('이 경고는 현재 최적화 계획의 현금여유를 평가한 결정론적 지표로, 현금 소진 확률을 뜻하지 않습니다. 진단용 평균 월 지출에는 신규 공장 투자비가 포함되며 매출 유입은 별도로 계산하지 않습니다.')
            st.markdown('<div class="section-title">시간 순서대로 보는 공급망 전환</div>',unsafe_allow_html=True)
            if st.session_state.replay_cache is None or st.session_state.get('replay_mode_cache') != (r['mode'],r.get('total_funding_MUSD')):
                with st.spinner('월별 공급망 통합 애니메이션 구성 중...'):
                    st.session_state.replay_cache=replay(r,exchange_rate=float(cfg['환율_KRW_USD']))
                    st.session_state.replay_mode_cache=(r['mode'],r.get('total_funding_MUSD'))
            st.plotly_chart(st.session_state.replay_cache,use_container_width=True,config={'displaylogo':False,'responsive':True})
            st.info('아래 차트의 ▶ 재생을 누르면 지도·조달물량·적격비율·통합 현금잔액이 한 화면에서 월별로 동시에 바뀝니다.')
            st.caption('재생 컨트롤과 월 슬라이더로 지도·조달량·비PFE 비율·통합 현금잔액을 함께 확인할 수 있습니다. 월별 근거는 아래에서 별도 선택합니다.')
            st.markdown('<div class="section-title">선택의 흐름 · 기업별 조달과 공장 건설 간트차트</div>',unsafe_allow_html=True)
            st.plotly_chart(decision_gantt(r),use_container_width=True,config={'displaylogo':False})
            st.caption('분홍색=비PFE 비적격 분류 · 파란색=적격 분류 · 보라색 윤곽=유지 중인 계약 · 주황색=공장 건설 · 초록색=가동. 마우스를 올리면 계약 유지기간과 공급량을 확인할 수 있습니다.')
            with st.expander('📋 월별 주요 의사결정 일지 · 전 기간',expanded=False):
                st.dataframe(decision_events(r),hide_index=True,use_container_width=True)
            st.markdown('<div class="section-title">월별 핵심 사건 및 공장 상태</div>',unsafe_allow_html=True)
            selected=st.select_slider('상세히 확인할 월',options=s['월'].tolist(),value=s.iloc[deadline]['월'])
            t=s.index[s['월']==selected][0]
            left,right=st.columns([1.25,1])
            with left:
                st.markdown(f"**{selected} 발생 이벤트**")
                for ev in events(r,t):st.markdown(f'<div class="event">{ev}</div>',unsafe_allow_html=True)
                if not events(r,t):st.info('이번 달 특별한 착공·가동 이벤트는 없습니다.')
                for v in status_text(r,t):st.caption('🏭 '+v)
            with left:
                active_contracts=r.get('contracts',pd.DataFrame())
                if not active_contracts.empty:
                    current=active_contracts[(active_contracts['t']==t)&(active_contracts['계약유지_1']==1)]
                    if not current.empty:
                        with st.expander(f'계약 유지 중인 공급기업 {len(current)}개 · 계약조건',expanded=False):
                            st.dataframe(current[['기업','최소계약기간_개월','계약최소구매율','신규계약_1']].rename(columns={'최소계약기간_개월':'최소 유지기간(월)','계약최소구매율':'월 최소구매율','신규계약_1':'이번 달 신규계약'}),hide_index=True,use_container_width=True)
            with right:
                k=s.iloc[t];cash=r['budgets'].iloc[t]
                a1,a2=st.columns(2)
                a1.metric('이번 달 비PFE 비율',f"{k['비PFE비율']*100:.1f}%")
                a2.metric('월말 회사 현금잔액',won_label(cash['월말현금잔액_MUSD']))
                a3,a4=st.columns(2)
                a3.metric('기업별 수입 총량',f"{r['shipments'].query('t == @t')['수입량_톤'].sum():,.0f}톤")
                a4.metric('자체 생산량',f"{k['자체생산_톤']:,.0f}톤")
            st.markdown('<div class="section-title">해당 월의 선택 근거 · 왜 이 기업에서 수입하고 이 공장을 건설했나?</div>',unsafe_allow_html=True)
            st.dataframe(why_month(r,int(t)),hide_index=True,use_container_width=True)
            st.caption('선택 근거는 조달 물량·단가·공급상한·통합 현금잔액 기준이며, 대안 선택의 인과 비교는 별도 재최적화로 확인합니다.')
            st.markdown('<div class="section-title">규제 달성 시점 · 유지비용</div>',unsafe_allow_html=True)
            # A date observed in an optimized plan must never be labelled as the globally earliest
            # achievable date. It is determined directly from the SAME plan's monthly results.
            summary_months=s['월'].tolist()
            ratios=s['비PFE비율'].to_numpy(dtype=float)
            reference_targets=s['법률참고목표'].to_numpy(dtype=float)
            demand_met=s['미공급_톤'].to_numpy(dtype=float)<1e-5
            if r['mode']=='earliest':
                flags=r.get('flags')
                candidate_idx=np.where(flags>0.5)[0] if flags is not None else np.array([],dtype=int)
                first_stable_idx=int(candidate_idx[0]) if len(candidate_idx) else None
                date_kind='가장 빠른 지속 달성월 (탐색결과)'
            elif r['mode'] in ('strict','flex_budget'):
                deadline_goal=float(reference_targets[deadline])
                # For pre-deadline attainment, use at least the selected deadline's threshold.
                # After deadline, enforce the varying target applicable in each future month.
                threshold=np.maximum(reference_targets,deadline_goal)
                compliant=(ratios+1e-6>=threshold)&demand_met
                future_all=np.logical_and.accumulate(compliant[::-1])[::-1]
                possible=np.flatnonzero(future_all)
                first_stable_idx=int(possible[0]) if len(possible) else None
                date_kind='선택된 조달계획의 최초 지속 충족월'
            else:
                first_stable_idx=None
                date_kind='지속 달성월'
            col1,col2=st.columns([1,1],gap='large')
            with col1:
                if first_stable_idx is not None:
                    st.metric(date_kind,summary_months[first_stable_idx])
                    if r['mode']=='earliest':
                        if r.get('optimal',False):
                            st.success('해당 예산조건에서 MILP가 가장 빠른 지속 달성 시점을 확인했습니다.')
                        else:
                            st.warning('발견한 지속 달성 시점입니다. 최단시점의 최적성은 증명되지 않았습니다.')
                    else:
                        st.success('표시 중인 계획의 월별 실제 조달비율을 검증한 날짜입니다.')
                        st.caption('이는 선택된 계획에서 처음으로 목표 수준을 달성하고 이후에도 유지한 달입니다. 가능한 모든 대안 중 가장 빠른 달을 뜻하지는 않습니다.')
                elif r['mode']=='diagnostic':
                    st.warning('이 결과는 규제 미달성 보조계획입니다. 지속 달성 시점은 확인되지 않았습니다.')
                else:
                    st.info('표시 중인 계획에서는 지속 달성 시작월을 확인하지 못했습니다. 분석 조건과 솔버 상태를 점검하세요.')
                if r['mode']=='diagnostic':
                    st.metric('목표월 실제 비PFE 조달비율',f"{sgoal['비PFE비율']*100:.1f}%")
                    st.metric('목표월 기준 부족분',f"{max(0,(sgoal['법률참고목표']-sgoal['비PFE비율'])*100):.1f}%p")
                    st.caption('보조해 결과로 계산된 부족분이며 수학적으로 달성 가능한 최대비율과 같다고 보장하지 않습니다.')
            with col2:
                upkeep=funds_to_maintain(r)
                if upkeep:
                    st.metric('목표 이후 월평균 조달·운영비',won_label(upkeep['평균월조달운영비_MUSD']))
                    st.metric('목표 이후 최대 월 조달·운영비',won_label(upkeep['최대월조달운영비_MUSD']))
                    st.caption('선택된 계획의 구매·생산·고정비입니다. 신규 공장 CAPEX 및 매출은 제외되며 총 자금조달액과 다릅니다.')
            st.markdown('<div class="section-title">최적 선택의 경제적 근거</div>',unsafe_allow_html=True)
            tabs=st.tabs(['🏭 공장 선택 근거','🚢 기업별 조달 근거','💵 비용 구조','🔎 계산 검증'])
            with tabs[0]:
                if df.empty:st.info('최적해에서 공장 건설을 선택하지 않았습니다. 기존 조달만으로 목표를 충족할 수 있는 상태입니다.')
                else:
                    krwdf=df[['국가','공장','착공월','가동월','건설비_MUSD']].copy();krwdf['건설비_억원']=krwdf.pop('건설비_MUSD').map(krw);st.dataframe(krwdf,hide_index=True,use_container_width=True)
                    st.markdown('선택 기업과 국가의 비용·공기·원료·예산 제약을 동시에 만족하는 후보 중 총비용이 작은 조합을 찾았습니다. 다른 국가가 더 비싸다는 단정은 대안 재최적화로 검증해야 합니다.')
                    if st.button('국가별 공장 최소 1개 강제 건설 후 비용 비교'):
                        rows=[]
                        for country in D['plants']['국가']:
                            alt=solve(r['data'],mode=('earliest' if r['mode']=='earliest' else 'flex_budget' if r['mode']=='flex_budget' else 'strict'),force_country=country,time_limit=30)
                            flags=alt.get('flags')
                            first_idx=np.where(flags>0.5)[0] if flags is not None else []
                            start_month=alt['summary'].iloc[int(first_idx[0])]['월'] if len(first_idx) else ('기간 내 불가' if flags is not None else '기한 준수' if alt['feasible'] else '불가')
                            rows.append({'대안':country,'계산 가능해':alt['feasible'],'지속 달성월':start_month,'할인 총비용(MUSD)':round(alt['total_cost_MUSD'],2) if alt['feasible'] else np.nan,'비용 차이(MUSD)':round(alt['total_cost_MUSD']-r['total_cost_MUSD'],2) if alt['feasible'] and r['mode']=='strict' else np.nan,'외부조달액 차이(MUSD)':round(alt['total_funding_MUSD']-r['total_funding_MUSD'],2) if alt['feasible'] and r['mode']=='flex_budget' else np.nan})
                        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                        st.caption('동일한 통합예산·규제 조건에서 국가별 최소 1개 공장 건설을 강제해 재최적화. 최초 달성월이 다른 대안끼리는 단순 비용차로 우위를 판단할 수 없습니다. 계산시간 제한 내 최적성 보장은 별도로 확인해야 합니다.')
            with tabs[1]:
                costs=r['shipments'].groupby(['국가','기업','비PFE_1'],as_index=False).agg({'수입량_톤':'sum','수입비_MUSD':'sum'})
                costs=costs.sort_values('수입량_톤',ascending=False)
                st.dataframe(costs,hide_index=True,use_container_width=True)
                st.caption('가격·공급상한·공급계약기간·PFE 구분·공장 생산능력을 함께 반영합니다.')
            with tabs[2]:
                st.plotly_chart(cost_breakdown(r,exchange_rate=float(cfg['환율_KRW_USD'])),use_container_width=True)
                money_df=s[['월','수입비_MUSD','운영비_MUSD','건설비_MUSD','총지출_MUSD']].copy()
                for money_col in list(money_df.columns)[1:]:money_df[money_col.replace('_MUSD','_억원')]=money_df.pop(money_col).map(krw)
                st.dataframe(money_df,hide_index=True,use_container_width=True)
            with tabs[3]:
                st.write('수요 미공급량:',round(float(s['미공급_톤'].sum()),5),'톤')
                st.write('회사 현금 최소 잔액:',round(float(r['budgets']['월말현금잔액_MUSD'].min()),4),'백만 USD')
                st.write('목표 이후 미충족 월수:',int((~s.loc[deadline:,'목표충족']).sum()))
                st.write('솔버 상태:',r['message'])
                st.caption('원료·가동능력·공기·공급계약·예산 제약을 모델에 적용합니다.')
            a,b=st.columns(2)
            with a:st.download_button('최적화 결과 Excel 다운로드',excel_result_bytes(r),'LiOH_월별_최적화_결과.xlsx')
            with b:st.download_button('애니메이션 · 타임라인 · 엑셀 패키지 다운로드',output_zip(r),'LiOH_STRATEGY_결과패키지.zip',type='primary')
            st.caption('마지막 분석 실행: '+st.session_state.last_exec+' · 데이터 입력이 변경되면 재실행해야 결과가 갱신됩니다.')
