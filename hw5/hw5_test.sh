wget 'https://www.dropbox.com/s/d6oc3gcjusav8sd/model.0002-0.9961.h5?dl=0' -O bestModel
python predict.py $1 $2 $3
