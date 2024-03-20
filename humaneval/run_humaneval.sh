path=".outputs/Llama-2-7b.json"    #文件地址
# path='./static_ptq_1-2t_smooth_clip.jsonl'
field="pred_output"   #字段名
# field='infer_out'

python humaneval.py --path $path --field $field
