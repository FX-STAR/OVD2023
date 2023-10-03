

python submit/ovdet2023/wbf_fg.py


num_p=16
for((i=0;i<($num_p);i++));  
do
CUDA_VISIBLE_DEVICES=$(($i%8)) python submit/ovdet2023/cn_infer_HL.py $i $num_p &
done 

wait

python submit/ovdet2023/cn_merge.py