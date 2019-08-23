# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:01:12 2019

@author: lisong
"""

import pandas as pd
import pymysql
from sqlalchemy import create_engine
from WindPy import *
from datetime import datetime
"""
pd.DataFrame.to_sql()可以使用sqlalchemy，也可以指定格式。在建表的问题可以解决
合并数据库，可以使用内存，无需直接使用sql。利用pandas对数据清洗合并，并在导入数据库方便调用。
cur.execute(""
            select column_name 
            from information_schema.columns 
            where table_name='temp' and table_schema='mutual_fund_schema'")
pd.read_sql('select * from temp',connect)

s='mysql+pymysql://root:root@localhost:3306/mutual_fund_schema'

w.wss("001619.OF", "issue_date,issue_totalunit,fund_setupdate","unit=1")
"""
def Fund_wind_code(date):
    parameter="date={0};sectorid=2001000000000000".format(date)
    e,r=w.wset("sectorconstituent",parameter,usedf=True)#get all Mutual Fund wind_code
    return r.set_index('wind_code')
#下面代码已实现数据导出
def Fund_unit (date,d,indicator="unit_total,fund_investtype,fund_firstinvesttype,fund_custodianbank",chunksize=3500): 
    parameter="unit=1;tradeDate="+date
    size = (len(d)//chunksize)+1
    r=list(range(size))
    for i in r:
        if chunksize*(i+1)>len(d):
            e,r[i]=w.wss(d[chunksize*i:], indicator,parameter,usedf=True)
        else:
            e,r[i]=w.wss(d[chunksize*i:chunksize*(i+1)], indicator,parameter,usedf=True) 
    return pd.concat(r)

def Fund_Issue (indicator="issue_date,issue_totalunit,fund_setupdate",chunksize=3500): 
    cur=My_sql_connect()
    cur.execute('select `wind_code` from fund_unit')
    d=[i[0] for i in cur.fetchall()]
    size = (len(d)//chunksize)+1
    r=list(range(size))
    w.start()
    for i in r:
        if chunksize*(i+1)>len(d):
            e,r[i]=w.wss(d[chunksize*i:], indicator,"unit=1",usedf=True)
        else:
            e,r[i]=w.wss(d[chunksize*i:chunksize*(i+1)], indicator,"unit=1",usedf=True)
    w.close()
    ret=pd.concat(r)
    ret.columns=[str.lower(r) for r in ret.columns]
    ret.index.name='wind_code'
    ret.reset_index(inplace=True)
    ret['issue_date']=pd.to_datetime(ret['issue_date'])
    ret['fund_setupdate']=pd.to_datetime(ret['fund_setupdate'])
    return {ret.columns[1]:ret}

def Trans_date_to_quarter(date):
    quarter_delta= pd.Period(datetime.now().strftime('%Y-%m-%d'),'Q-DEC')-pd.Period(date,'Q-DEC')
    if quarter_delta.n==0 :
        return pd.Period(datetime.now().strftime('%Y-%m-%d'),'q-dec')-1
    elif quarter_delta.n < 0:
        print('Your enter date is too forward ,there is no data of your date')
    else:
        return pd.Period(pd.Period(date)+1,'q-dec')-1

#要完成数据的切片和使用。
def Data_format(date):#Download data from wind by the 'date'.
    w.start()
    code=Fund_wind_code(date)
    d=list(code.index)
    unit = Fund_unit(date,d)
    w.close()
    quarter_trans = Trans_date_to_quarter(date)
    unit.index.name=code.index.name
    unit.columns=[str.lower(r) for r in unit.columns]
    fund_secname=code['sec_name'].reset_index()
    fund_unit = unit['unit_total'].reset_index()
    fund_unit.rename(columns={'unit_total':quarter_trans},inplace=True)
    fund_custodianbank = unit['fund_custodianbank'].reset_index()
    fund_investtype=unit[['fund_investtype','fund_firstinvesttype']].reset_index()
    print('The value of the got_list is {fund_secname,fund_unit,fund_custodianbank,fund_investtype}')
    return {'sec_name':fund_secname,
            'fund_unit':fund_unit,
            'fund_custodianbank':fund_custodianbank,
            'fund_investtype':fund_investtype}

def Load_database(grids):
    #grids= Data_format()'s value
    s='mysql+pymysql://root:root@localhost:3306/mutual_fund_schema'
    engine =create_engine(s)
    for i in grids.keys():
        grids[i].to_sql(i,engine,schema='mutual_fund_schema',index=False,if_exists = 'replace')

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

def Fetch_database(grids_keys):#Download data from mysql by now.
    #grids= Data_format()'s keys or other list.
    cur=My_sql_connect()
    #1.读取表格，2.为表格设置代码。
    fetch_sql_list=['select * from mutual_fund_schema.{0}'.format(i) for i in grids_keys]
    tables=[]
    for sql in fetch_sql_list:
        cur.execute(sql)
        result=list(cur.fetchall())
        column = [i[0] for i in cur.description]
        tables.append(pd.DataFrame(result,columns=column))
    t={} 
    for tab in tables:
        if tab.columns[1][0] != '2':
            t[tab.columns[1]] = tab
        else:
            t['fund_unit']=tab
    return t

def Merge_grids(new_tables,old_tables):#Merge wind's data & mysql's data into new tables in rom
    tables={}
    tables['fund_unit']=pd.merge(new_tables['fund_unit'],
              old_tables['fund_unit'],how='outer',on='wind_code')
    for k in new_tables.keys():
        if k not in tables.keys():
            tables[k]=pd.concat([new_tables[k],old_tables[k]],
                      axis=0,ignore_index=True,join='outer').drop_duplicates('wind_code')
    return tables

def Regular_update(date):
    new_dic=Data_format(date)#Download data from wind by the 'date'.
    old_dic=Fetch_database(list(new_dic.keys()))#Download data from mysql by now.
    merge_dic=Merge_grids(new_dic,old_dic)#Merge wind's data & mysql's data into new tables in rom
    Load_database(merge_dic)#Load new tables to mysql
    Load_database(Fund_Issue())#Accrording to new wind_code,fetch their Issue affairs and reload to mysql
"""
之所以只有2014Q3至今的数据，是因为我一开始只有2014Q3的数据，这里的代码都是更新，根据最新的日期更新
但是，使用起来仔细想想还有很多槽点。但是即使如此，自己会用就行了。
"""
  
date_list=['20180801','20180501','20180201','20171101',
           '20170801','20170501','20170201','20161101',
           '20160801','20160501','20160201','20151101',
           '20150801','20150501','20150201','20141101']