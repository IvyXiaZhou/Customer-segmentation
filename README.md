# Customer-segmentation
## 客户分层项目
## 项目介绍
本项目基于某零售企业会员信息与销售流水数据，通过数据清洗、探索性分析、用户画像构建及聚类算法，实现会员消费行为深度洞察与客户分层，为精准营销和客户关系管
理提供数据支撑。
## **数据来源**
数据集百度云下载方式 
链接: https://pan.baidu.com/s/12W4Bi8gkYNaEDTi6M9o_gA 提取码: 3xd9
会员信息表（cumcm2018c1.xlsx）：包含会员卡号、性别、出生日期、登记时间等基础信息
销售流水表（cumcm2018c2.csv）：包含消费时间、消费金额、销售数量、会员积分等交易数据
## 输出文件说明
文件路径	     说明
./vip_info.csv 清洗后的会员消费详细数据
./LRFMPSX.csv 会员LRFMPSX核心指标与特征数据
./consumers_profile.csv	会员个性化画像数据
./ 会员出生年代及男女比例情况.png 会员基础特征可视化图表
./ 季度和天数的均值偏好情况.png	消费时间偏好可视化图表
./LRFMP 聚类轮廓系数图.png	聚类效果评估图表
./ 两类客户的 LRFMP 均值差异.png	客户分层差异可视化图表
## 运行说明
环境依赖安装
bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn openpyxl
数据准备：将cumcm2018c1.xlsx和cumcm2018c2.csv放入项目根目录
执行主脚本：直接运行代码文件，自动生成所有分析结果与可视化图表
## 核心结论
会员以中年女性群体为主，消费高峰集中在特定季度与时段，存在明显时间偏好；本项目通过 KMeans 聚类将会员划分为两类群体，他们在消费频次、金额与积分上存在显著
差异。
## 应用建议
基于客户分层结果，分别为高价值和一般价值客户制定差异化运营策略，并结合用户画像实现精准推送，提升营销转化率。

## Customer Segmentation Project
## Project Introduction
This project is based on member information and sales transaction data of a department store. Through data cleaning, exploratory analysis, user profile construction and clustering algorithms, it achieves in-depth insight into member consumption behavior and customer segmentation, providing data support for precision marketing and customer relationship management.
## **Data Sources**
Baidu Cloud download link for the dataset:
Link: https://pan.baidu.com/s/12W4Bi8gkYNaEDTi6M9o_gA  Extraction Code: 3xd9
Member Information Table (cumcm2018c1.xlsx): Contains basic information such as member ID, gender, date of birth, and registration time.
Sales Transaction Table (cumcm2018c2.csv): Contains transaction data such as consumption time, consumption amount, sales quantity, and member points.
## Output File Description
File Path	      Description
./vip_info.csv	Cleaned detailed member consumption data
./LRFMPSX.csv	Core indicators and feature data of member LRFMPSX
./consumers_profile.csv	Personalized member profile data
./Member Birth Decade and Gender Ratio.png	Visual chart of basic member characteristics
./Quarterly and Daily Average Consumption Preference.png	Visual chart of consumption time preference
./LRFMP Clustering Silhouette Coefficient Chart.png	Clustering effect evaluation chart
./LRFMP Average Difference Between Two Customer Groups.png	Visual chart of customer segmentation differences
## Running Instructions
Environment Dependence Installation
bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn openpyxl
Data Preparation
Place cumcm2018c1.xlsx and cumcm2018c2.csv in the project root directory.
Execute the Main Script
Run the code file directly to automatically generate all analysis results and visual charts.
## Core Conclusions
Members are mainly middle-aged females, with consumption peaks concentrated in specific quarters and time periods, showing obvious time preferences. This project divides members into two groups through KMeans clustering, and there are significant differences in their consumption frequency, amount and points.
## Application Suggestions
Based on the customer segmentation results, formulate differentiated operation strategies for high-value and general-value customers respectively, and combine user profiles to achieve precise push and improve marketing conversion rate.
