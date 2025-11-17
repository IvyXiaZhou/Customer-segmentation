import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'font.size':16})
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

df_cum=pd.read_excel('./cumcm2018c1.xlsx')
df_sale=pd.read_csv('./cumcm2018c2.csv')

df_cum.drop_duplicates(subset='会员卡号',inplace=True)
print('会员卡号（去重）有{}条记录'.format(len(df_cum['会员卡号'].unique())))
df_cum.dropna(subset=['登记时间'],inplace=True)
print('df_cum(去重和去缺失)有{}条记录'.format(df_cum.shape[0]))
df_cum['性别'].fillna(df_cum['性别'].mode().values[0],inplace=True)
df_cum.info()
print(df_cum.isnull().mean())
L=pd.DataFrame(df_cum.loc[df_cum['出生日期'].notnull(),['出生日期','性别']])
L['年龄']=L['出生日期'].astype(str).apply(lambda x:x[:3]+'0')
L.drop('出生日期',axis=1,inplace=True)
L['年龄']=L['年龄'].astype(int)
condition="年龄>=1920 and 年龄<=2020"
L=L.query(condition)
L.index=range(L.shape[0])
print(L['年龄'].value_counts())
df_cum.drop('出生日期',axis=1,inplace=True)
df_cum.index=range(df_cum.shape[0])
print('数据清洗之后共有{}行记录，{}列字段，字段名为{}'.format(df_cum.shape[0],df_cum.shape[1],df_cum.columns))
print(df_cum)

print(df_sale.columns)
df_sale.info()
print('销售数量大于0的记录有：{}\t 全部记录有：{}\t 两者是否相等：{}'.format(len(df_sale['销售数量'] > 0),
                                                     df_sale.shape[0], len(df_sale['销售数量'] > 0) == df_sale.shape[0]))
print('消费金额d大于0的记录有：{}\t 全部记录有：{}\t 两者是否相等: {}'.format(len(df_sale['消费金额']>0),
                df_sale.shape[0],len(df_sale['消费金额']>0)==df_sale.shape[0]))
df_sale_clean=df_sale.dropna(subset=['会员卡号'])
print(df_sale_clean.isnull().mean())
df_sale_clean.drop(['收银机号','柜组编码','柜组名称'],axis=1,inplace=True)
df_sale_clean.index=range(df_sale_clean.shape[0])
print(type(df_sale_clean)==type(df_cum))
print(f'会员信息表中的记录为{len(df_cum)}\t销售流水表中的记录为{len(df_sale_clean)}')
df=pd.merge(df_sale_clean,df_cum,on='会员卡号',how='left')
index1=df['消费金额']>0
index2=df['此次消费的会员积分']>0
index3=df['销售数量']>0
df1=df.loc[index1&index2&index3,:]
df1.index=range(df1.shape[0])
df1['会员']=1
df1.loc[df1['性别'].isnull(),'会员']=0
print(df1.head())
L['性别']=L['性别'].apply(lambda x:'男' if x==1 else '女')
gen=L['性别'].value_counts()
L['年龄段']='中年'
L.loc[L['年龄']<=1950,'年龄段']='老年'
L.loc[L['年龄']>=1990,'年龄段']='青年'
res=L['年龄段'].value_counts()

fig,axs=plt.subplots(1,2,figsize=(16,7),dpi=100)
ax=sns.countplot(x='年龄',data=L,ax=axs[0])
for p in ax.patches:
    height=p.get_height()
    ax.text(x=p.get_x()+(p.get_width()/2),y=height+500,s='{:.0f}'.format(height),ha='center')
axs[0].set_title('会员的出生年代')
axs[1].pie(gen,labels=gen.index,wedgeprops={'width':0.4},counterclock=False,autopct='%.2f%%',pctdistance=0.8)
axs[1].set_title('会员的男女比例')
plt.savefig('./会员出生年代及男女比例情况.png')
plt.figure(figsize=(8,6),dpi=100)
plt.pie(res.values,labels=['中年','青年','老年'],autopct='%.2f%%',pctdistance=0.8,
        counterclock=False,wedgeprops={'width':0.4})
plt.title('会员的年龄分布')
plt.savefig('./会员的年龄分布.png')
fig,axs=plt.subplots(1,2,figsize=(12,7),dpi=100)
axs[0].pie([len(df1.loc[df1['会员']==1,'消费产生的时间'].unique()),len(df1.loc[df1['会员']==0,'消费产生的时间'].unique())],
           labels=['会员','非会员'],wedgeprops={'width':0.4},counterclock=False,autopct='%.2f%%',pctdistance=0.8)
axs[0].set_title('总订单占比')
axs[1].pie([df1.loc[df1['会员']==1,'消费金额'].sum(),df1.loc[df1['会员']==0,'消费金额'].sum()],
           labels=['会员','非会员'],wedgeprops={'width':0.4},counterclock=False,autopct='%.2f%%',pctdistance=0.8)
axs[1].set_title('总消费金额占比')
plt.savefig('./总订单和总消费占比情况.png')

df_vip=df1.dropna()
df_vip.drop(['会员'],axis=1,inplace=True)
df_vip.index=range(df_vip.shape[0])
df_vip['消费产生的时间'] = pd.to_datetime(df_vip['消费产生的时间'])
df_vip['年份']=df_vip['消费产生的时间'].dt.year
df_vip['月份']=df_vip['消费产生的时间'].dt.month
df_vip['季度']=df_vip['消费产生的时间'].dt.quarter
df_vip['天']=df_vip['消费产生的时间'].dt.day

def orders(df,label,div):
    x_list=np.sort(df[label].unique().tolist())
    order_nums=[]
    for i in range(len(x_list)):
        order_nums.append(int(len(df.loc[df[label]==x_list[i],'消费产生的时间'].unique())/div))
    return x_list,order_nums

quarters_list,quarters_order=orders(df_vip,'季度',3)
days_list,days_order=orders(df_vip,'天',36)
time_list=[quarters_list,days_list]
order_list=[quarters_order,days_order]
maxindex_list=[quarters_order.index(max(quarters_order)),days_order.index(max(days_order))]
fig,axs=plt.subplots(1,2,figsize=(18,7),dpi=100)
colors=np.random.choice(['r','g','b','orange','y'],replace=False,size=len(axs))
titles=['季度的均值消费偏好','天数的均值消费偏好']
labels=['季度','天数']
for i in range(len(axs)):
    ax=axs[i]
    ax.plot(time_list[i],order_list[i],linestyle='-.',c=colors[i],marker='o',alpha=0.85)
    ax.axvline(x=time_list[i][maxindex_list[i]],linestyle='--',c='k',alpha=0.8)
    ax.set_title(titles[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel('均值消费订单数')
    print(f'{titles[i]}最优的时间为: {time_list[i][maxindex_list[i]]}\t 对应的均值消费订单数为：{order_list[i][maxindex_list[i]]}')
plt.savefig('./季度和天数的均值偏好情况.png')

def plot_qd(df,label_y,label_m,nrow,ncol):
    y_list=np.sort(df[label_y].unique().tolist())[:-1]
    colors=np.random.choice(['r','g','b','orange','y','k','c','m'],replace=False,size=len(y_list))
    markers=['o','^','v']
    plt.figure(figsize=(8,6),dpi=100)
    fig,axs=plt.subplots(nrow,ncol,figsize=(16,7),dpi=100)
    for k in range(len(label_m)):
        m_list=np.sort(df[label_m[k]].unique().tolist())
        for i in range(len(y_list)):
            order_m=[]
            index1=df[label_y]==y_list[i]
            for j in range(len(m_list)):
                index2=df[label_m[k]]==m_list[j]
                order_m.append(len(df.loc[index1 & index2,'消费产生的时间'].unique()))
            axs[k].plot(m_list,order_m,linestyle='-.',c=colors[i],alpha=0.8,marker=markers[i],label=y_list[i],markersize=4)
        axs[k].set_xlabel(f'{label_m[k]}')
        axs[k].set_ylabel('消费订单数')
        axs[k].set_title(f'2015-2018年会员的{label_m[k]}消费订单差异')
        axs[k].legend()
    plt.savefig(f'./2015-2018年会员的{"和".join(label_m)}消费订单差异.png')
plot_qd(df_vip,'年份',['季度','天'],1,2)

def plot_ym(df,label_y,label_m):
    y_list=np.sort(df[label_y].unique().tolist())[:-1]
    m_list=np.sort(df[label_m].unique().tolist())
    colors=np.random.choice(['r','g','b','orange','y'],replace=False,size=len(y_list))
    markers=['o','^','v']
    fig,axs=plt.subplots(1,2,figsize=(18,8),dpi=100)
    for i in range(len(y_list)):
        order_m=[]
        money_m=[]
        index1=df[label_y]==y_list[i]
        for j in range(len(m_list)):
            index2=df[label_m]==m_list[j]
            order_m.append(len(df.loc[index1 & index2, '消费产生的时间'].unique()))
            money_m.append(df.loc[index1 & index2, '消费金额'].sum())
        axs[0].plot(m_list,order_m,linestyle='-.',c=colors[i],alpha=0.8,marker =markers[i],label=y_list[i])
        axs[1].plot(m_list,order_m, linestyle='-.', c=colors[i],alpha=0.8, marker =markers[i],label=y_list[i])
        axs[0].set_xlabel('月份')
        axs[0].set_ylabel('消费订单数')
        axs[0].set_title('2015-2018年会员的消费订单差异')
        axs[1].set_xlabel('月份')
        axs[1].set_ylabel('消费金额总数')
        axs[1].set_title('2015-2018年会员的消费金额差异')
        axs[0].legend()
        axs[1].legend()
    plt.savefig('./2015-2018年会员的消费订单和金额差异.png')
plot_ym(df_vip,'年份','月份')

df_vip['时间']=df_vip['消费产生的时间'].dt.hour
x_list,order_nums=orders(df_vip,'时间',1)
maxindex=order_nums.index(max(order_nums))
plt.figure(figsize=(8,6),dpi=100)
plt.plot(x_list,order_nums,linestyle='-.',marker='o',c='m',alpha=0.8)
plt.xlabel('小时')
plt.ylabel('消费订单')
plt.axvline(x=x_list[maxindex],linestyle='--',c='r',alpha=0.6)
plt.title('2015-2018年各段小时的销售订单数')
plt.savefig('./2015-2018年各段小时的销售订单数.png')
df_vip.to_csv('./vip_info.csv',encoding='gb18030',index=None)

df_vip=pd.read_csv('./vip_info.csv',encoding='gbk')
df_vip.info()
print('消费产生的时间存在异常值的数量为：{}\t 登记时间存在异常值数量为： {}'.format(len(df_vip[df_vip['消费产生的时间']>='2018-01-03']),len(df_vip[df_vip['登记时间']>='2018-01-03'])))
index1=df_vip['消费产生的时间']<'2018-01-03'
index2=df_vip['登记时间']<'2018-01-03'
df_vip=df_vip[index1 & index2]
df_vip.index=range(df_vip.shape[0])
'''print('筛选全部异常值之后数据的记录数为：{}\t共有{}个字段'.format(df_vip.shape[0],df_vip.shape[1]))
print('会员总数：{}\t 记录数: {}'.format(len(df_vip['会员卡号'].unique()),df_vip.shape[0]))
print(df_vip.columns)'''

def time_minus(df,end_time):
    df.columns=['A','B']
    df['C']=end_time
    l=pd.to_datetime(df['C'])-pd.to_datetime(df['B'])
    l=l.apply(lambda x: str(x).split(' ')[0])
    l=l.astype(int)/30
    return l
df_L=df_vip.groupby('会员卡号')['登记时间'].agg(lambda x:x.values[-1]).reset_index()
df_R=df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x:x.values[-1]).reset_index()
end_time='2018-1-3'
L=time_minus(df_L,end_time)
R=time_minus(df_R,end_time)

F=df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x: len(np.unique(x.values))).reset_index(drop=True)
M=df_vip.groupby('会员卡号')['消费金额'].agg(lambda x: np.sum(x.values)).reset_index(drop=True)
P=df_vip.groupby('会员卡号')['此次消费的会员积分'].agg(lambda x: np.sum(x.values)).reset_index(drop=True)
df_vip['消费时间偏好']=df_vip['时间'].apply(lambda x:'晚上'if x>=18 else '下午'if x>=14 else'中午'
                            if x>=11 else '上午' if x>=6 else '凌晨')
#print(df_vip)
S=df_vip.groupby('会员卡号')['消费时间偏好'].agg(lambda x: x.mode().values[0]).reset_index(drop=True)
X=df_vip.groupby('会员卡号')['性别'].agg(lambda x: '女'if x.unique()[0]==0 else '男').reset_index(drop=True)
df_i=pd.Series(df_vip['会员卡号'].unique())
df_LRFMPSX = pd.concat([df_i,L,R,F,M,P,S,X],axis=1)
df_LRFMPSX.columns=['id','L','R','F','M','P','S','X']
print(df_LRFMPSX.head())
df_LRFMPSX.to_csv('./LRFMPSX.csv',encoding='gb18030',index=None)

df=pd.read_csv('./LRFMPSX.csv',encoding='gbk')
print(f'数据集的shape:{df.shape}')
print(df.isnull().mean())
print(df.describe())
df_profile=pd.DataFrame()
df_profile['会员卡号']=df['id']
df_profile['性别']=df['X']
df_profile['消费偏好']=df['S'].apply(lambda x:'您喜欢在'+str(x)+'时间进行消费')
df_profile['入会程度']=df['L'].apply(lambda x:'老用户'if int(x)>=13 else '中等用户'if int(x)>=4 else '新用户')
df_profile['最近购买的时间']=df['R'].apply(lambda x:'您最近'+str(int(x)*30)+'天前进行过一次购物')
df_profile['消费频次']=df['F'].apply(lambda x:'高频消费'if x>=20 else '中频消费'if x>=6 else '低频消费')
df_profile['消费金额']=df['M'].apply(lambda x:'高等消费用户'if int(x)>=1e+05 else '中等消费用户'if int(x)>=1e+04 else '低等消费用户')
df_profile['消费积分']=df['P'].apply(lambda x:'高等积分用户'if int(x)>=1e+05 else '中等积分用户'if int(x)>=1e+04 else '低等积分用户')
print(df_profile)
df_profile.to_csv('./consumers_profile.csv',encoding='gb18030',index=None)
    
def wc_plot(df,id_label=None):
    myfont='C:/windows/Fonts/simkai.ttf'
    if id_label==None:
        id_label=df.loc[np.random.choice(range(df.shape[0])),'会员卡号']
    text=df[df['会员卡号']==id_label].T.iloc[:,0].values.tolist()
    plt.figure(dpi=100)
    wc=WordCloud(font_path=myfont,background_color='white',width=500,height=400).generate_from_text(' '.join(text))
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig(f'./会员卡号为{id_label}的用户画像.png')
    plt.show()
wc_plot(df_profile,'8527d4d0')
wc_plot(df_profile)

df0 = df.iloc[:,1:6]
res_std = StandardScaler().fit_transform(df0)

n_clusters = range(2,7)
scores = []
for i in range(len(n_clusters)):
    clf=KMeans(n_clusters=n_clusters[i],random_state=20).fit(res_std)
    scores.append(silhouette_score(res_std,clf.labels_))
maxindex=scores.index(max(scores))
plt.figure(figsize=(8,6),dpi=100)
plt.plot(n_clusters,scores,linestyle='-.',c='b',alpha=0.6,marker='o')
plt.axvline(x=n_clusters[maxindex],linestyle='--',c='r',alpha=0.5)
plt.title('LRFMP的聚类轮廓系数图')
plt.ylabel('silhouette_score')
plt.xlabel('n_clusters')
plt.savefig('./LRFMP聚类轮廓系数图.png')

def plot(features,clf_list,nrow,ncol,title):
    N=len(features)
    angles=np.linspace(0,2*np.pi,N,endpoint=False)
    angles=np.concatenate([angles,[angles[0]]])
    features=np.concatenate([features,[features[0]]])
    fig=plt.figure(figsize=(14,14),dpi=100)
    for i in range(len(clf_list)):
        clf=clf_list[i]
        centers=clf.cluster_centers_
        ax=fig.add_subplot(nrow,ncol,i+1,polar=True)
        ax.set_thetagrids(angles*180/np.pi,features)
        colors=np.random.choice(['r','g','b','y','k','orange'],replace=False,size=len(centers))
        for j in range(len(centers)):
            values=np.concatenate([centers[j,:],[centers[j,:][0]]])
            ax.plot(angles,values,c=colors[j],alpha=0.6,linestyle='-.',label='类别'+str(j+1))
            ax.fill(angles, values, c=colors[j], alpha=0.2)
        ax.set_title(f'n_clusters={len(centers)}')
        ax.legend()
    plt.suptitle(title)
    plt.savefig(f'./{title}.png')

features=list('LRFMP')
res_std=StandardScaler().fit_transform(df0)
res_mm=MinMaxScaler().fit_transform(df0)
res=[res_std, res_mm]
titles=['标准化处理后的聚类雷达图','归一化处理后的聚类雷达图']
for i in range(len(res)):
    clf=[]
    for j in range(2,6):
        clf.append(KMeans(n_clusters=j,random_state=20).fit(res[i]))
    plot(features,clf,2,2,titles[i])

clf=KMeans(n_clusters=2,random_state=20).fit(res_std)
df0['labels']=clf.labels_
print(df0)
print(f"类别0所占比例为：{df0['labels'].value_counts().values[0]/df0.shape[0]} \t 类别1所占的比例为："
      f"{df0['labels'].value_counts().values[1]/df0.shape[0]}")
print(df0['labels'].value_counts())

L_avg=df0.groupby('labels').agg({'L':np.mean}).reset_index()
R_avg=df0.groupby('labels').agg({'R':np.mean}).reset_index()
F_avg=df0.groupby('labels').agg({'F':np.mean}).reset_index()
M_avg=df0.groupby('labels').agg({'M':np.mean}).reset_index()
P_avg=df0.groupby('labels').agg({'P':np.mean}).reset_index()

def plot_bar(df_list,nrow,ncol):
    fig, axs=plt.subplots(nrow,ncol,figsize=(2*(ncol+2),2.5),dpi=100)
    for i in range(len(axs)):
        ax=axs[i]
        df=df_list[i]
        ax.bar(df.iloc[:,0],df.iloc[:,1],color='m',alpha=0.4,width=0.5)
        for x,y in enumerate(df.iloc[:,1].tolist()):
            ax.text(x,y/2,'%.0f'%y,va='bottom',ha='center',fontsize=12)
        ax.set_xticks([0,1])
        ax.set_yticks(())
        ax.set_title(f'{df.columns[1]}')
    plt.suptitle('两类客户的LRFMP均值差异',y=1.1, fontsize=14)
    plt.savefig('./两类客户的LRFMP均值差异.png')
df_list=[L_avg,R_avg,F_avg,M_avg,P_avg]
plot_bar(df_list,1,5)

