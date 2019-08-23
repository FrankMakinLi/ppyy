# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:28:30 2019

@author: lisong
"""
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pymysql

def My_sql_connect(host='localhost',port=3306,user='root',passwd='root',db='mutual_fund_schema',charset='utf8'):
    #Encapsulation connect of My_sql
    connect = pymysql.Connect(
        host=host,
        port=port,
        user=user,
        passwd=passwd,
        db=db,
        charset=charset)
    return connect.cursor()
cur=My_sql_connect()
#获得指定时间段的切片，前提是df.columns是pd.period类型
#构造一个字典，用于装载不同的数据df，并方便导出成excel
current_quarter=pd.Period(datetime.now().strftime('%Y-%m-%d'),'Q-DEC')
data_dict={}
def issue_up(start,end,df):
    return df.loc[:,pd.period_range(start,end,freq='q-dec')]
#按投资类型汇总份额
cur.execute("""
            select a.`sec_name`,c.*,b.`fund_investtype` 
            from sec_name as a ,fund_unit as c,fund_investtype as b 
            where c.`wind_code`=b.`wind_code`and c.`wind_code`=a.`wind_code` 
            and a.`wind_code`=b.`wind_code`""")
re=pd.DataFrame(list(cur.fetchall()),columns=[i[0] for i in cur.description])
fund_transition=re[re['wind_code'].apply(lambda x:'!1' in x)]
re_grouped_sum=re.groupby('fund_investtype').sum()/100000000
re_grouped_sum=re_grouped_sum.sort_index(axis=1,ascending= False)
data_dict['re_grouped_sum']=re_grouped_sum
ab=[i for i in re_grouped_sum.columns]
#获得投资类型和风险等级对应关系，并按风险等级汇总份额
cur.execute('SELECT * FROM risk_type')
risk=pd.DataFrame(list(cur.fetchall()),columns=[i[0] for i in cur.description])
risk_class=pd.merge(risk,re_grouped_sum,left_on='fund_investtype',right_index=True)
risk_class_grouped_sum=risk_class.groupby('risk_type').sum()
data_dict['risk_class_grouped_sum']=risk_class_grouped_sum

#获得产品发行序列，并对数据做必要的整理，包括统一到较低的数量级，成立时间转为所属季度，去掉成立日期的异常值，并将成立时间转为int
cur.execute("""
            select a.*,b.`fund_investtype` 
            from issue_date as a,fund_investtype as b 
            where a.`wind_code`=b.`wind_code`""")
issue=pd.DataFrame(list(cur.fetchall()),columns=[i[0] for i in cur.description])
issue.fillna(0,inplace=True)
issue=pd.merge(risk,issue,left_on='fund_investtype',right_on='fund_investtype')
issue['issue_totalunit']=issue['issue_totalunit']/100000000
issue['fund_setup_quarter']=issue['fund_setupdate'].apply(pd.Period,args=('q-dec',))
issue['setup_days']=issue['fund_setupdate']-issue['issue_date']

#要去掉异常值
issue_days=issue[issue['setup_days']<timedelta(200)]
issue_days=issue_days[issue_days['setup_days']>timedelta(0)]
issue_days['setup_days']=issue_days['setup_days'].apply(lambda x :x.days)
#获得当季成立的具体基金，并用字典装载
setup_current_quarter=pd.merge(issue_days[issue_days['fund_setup_quarter']==current_quarter-1],re[['sec_name','wind_code']],how='left')
data_dict['setup_current_quarter']=setup_current_quarter

#历史成立天数平均值。
issue_mean_days=issue_days.groupby(['fund_setup_quarter','fund_investtype']).mean()
issue_mean_days=issue_mean_days.drop(columns='issue_totalunit').unstack().T.fillna(0)
issue_mean_days.index=issue_mean_days.index.get_level_values(1)
issue_mean_days_quarter=issue_up(start=ab[-1],end=ab[0],df=issue_mean_days)
data_dict['issue_mean_days_quarter']=issue_mean_days_quarter

#按类型：当期发行天数历史比较
issue_days_compare=pd.concat([issue_mean_days_quarter['2019Q1'],issue_mean_days_quarter.mean(axis=1)],keys='fund_investtype',axis=1)
issue_days_compare.columns=pd.Index([current_quarter,'mean_days'])
data_dict['issue_days_compare']=issue_days_compare

#风险等级历史成立天数平均值
issue_drisk=pd.merge(risk,issue_mean_days_quarter,left_on='fund_investtype',right_index=True)
issue_mean_days_of_risk=issue_drisk.groupby('risk_type').mean()
data_dict['issue_mean_days_of_risk']=issue_mean_days_of_risk

#每季度成立份额的时间序列。
issue_grouped_totalunit_by_investtype=issue.groupby(['fund_setup_quarter','fund_investtype']).sum().unstack().T.fillna(0)
issue_grouped_totalunit_by_investtype.index=issue_grouped_totalunit_by_investtype.index.get_level_values(1)
issue_totalunit_of_quarter=issue_up(start=ab[-1],end=ab[0],df=issue_grouped_totalunit_by_investtype)
issue_risk=pd.merge(risk,issue_totalunit_of_quarter,left_on='fund_investtype',right_index=True)
issue_risk=issue_risk.groupby('risk_type').sum()
data_dict['issue_totalunit_of_quarter']=issue_totalunit_of_quarter
data_dict['issue_risk']=issue_risk


"""
#成立数量的时间序列,已从近期报告中删除该指标。
issue_grouped_quantity_by_investtype=issue.groupby(['fund_setup_quarter','fund_investtype']).count()
issue_list=list(range(len(ab)))
for i in issue_list:
    issue_list[i] = issue_grouped_quantity_by_investtype.loc[(ab[i],),'wind_code']
    issue_list[i].index=issue_list[i].index.get_level_values(1)
    issue_list[i].name=pd.Period(ab[i],freq='q-dec')
issue_quantity_of_quarter=pd.concat(issue_list,axis=1,sort=False).fillna(0)
issue_q_risk=pd.merge(risk,issue_quantity_of_quarter,left_on='fund_investtype',right_index=True)
issue_quantity_of_risk=issue_q_risk.groupby('risk_type').sum()
data_dict['issue_quantity_of_risk']=issue_quantity_of_risk
"""
#清盘，2018Q4没有且2018Q3有

"""
#本期发行份额少的类型,已经获得本期发行份额的所有产品，已经历史发行序列，所以该指标暂时无用。
setup_this_quarter=issue_days.loc[issue_days['fund_setup_quarter']==current_quarter,]
failed=setup_this_quarter.loc[(setup_this_quarter['issue_totalunit']<2) & (setup_this_quarter['issue_totalunit']>0),]
failed_sec_name=re.loc[re['wind_code'].isin(failed['wind_code']),['wind_code','sec_name']]
failed=pd.merge(failed,failed_sec_name,on='wind_code')
setup_this_quarter=pd.merge(setup_this_quarter,re.loc[re['wind_code'].isin(setup_this_quarter['wind_code']),['wind_code','sec_name']],on='wind_code')
data_dict['setup_this_quarter']=setup_this_quarter
"""
#持续营销
sustain_sales=re_grouped_sum[ab[0]]-re_grouped_sum[ab[1]]-issue_totalunit_of_quarter[ab[0]]
data_dict['sustain_sales']=sustain_sales

def To_excel(data_dict):
    writer= pd.ExcelWriter(r'C:\Users\LISONG\Desktop\日常研究\季度报告\{0}季报\{0}Data.xlsx'.format(ab[0]))
    for i in data_dict.keys():
        data_dict[i].to_excel(writer,sheet_name=i)
    writer.save()
    