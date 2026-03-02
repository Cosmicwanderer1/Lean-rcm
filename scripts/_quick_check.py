import json
lines = open(r'D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl','r',encoding='utf-8').readlines()
for i in [0,1,2,len(lines)-1]:
    r = json.loads(lines[i])
    print(f'[{i}] src={r["reflection_source"]}, type={r["error_type"]}, thm={r["theorem_name"][:30]}, ref_len={len(r["reflection"])}')
r0 = json.loads(lines[0])
r100 = json.loads(lines[100])
print(f'Fields r0: {list(r0.keys())}')
print(f'Fields r100: {list(r100.keys())}')
for l in lines[:500]:
    r = json.loads(l)
    if 'template' in r['reflection_source']:
        print(f'Template fields: {list(r.keys())}')
        print(f'Template src={r["reflection_source"]}, ref_len={len(r["reflection"])}')
        break
else:
    print('No template found in first 500')
