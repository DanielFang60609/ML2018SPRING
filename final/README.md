Machine Learning (2018,Spring) Final Project

Team:NTU_r06921058_95度的顯卡/R06921058 方浩宇 B04501127 凌于凱 D06921025 王瀚緯 R06942119 林宗憲

Package:

keras=2.0.8 numpy pandas scipy sklearn librosa matplotlib


Test方式:
bash test.sh [output path]

Train方式:
先用bash data.sh
下載預先處理好的資料

然後bash train.sh [path of train.csv]

使用其他的Param Train的方式:
先使用bash data.sh下載檔案
然後使用bash download_dataset.sh下載Raw data
在src中的Train.py 修改參數
使用bash train.sh [path of train.csv]

