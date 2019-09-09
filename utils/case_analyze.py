import ujson as json


def load_json(path):
    with open(path, 'r') as fh:
        error = json.load(fh)
    return error


mcdn_path = '../outputs/bootstrapped/MCDN/results/FALSE_valid.json'
scrn_path = '../outputs/bootstrapped/SCRN/results/FALSE_valid.json'
tb_path = '../outputs/bootstrapped/TB/results/FALSE_valid.json'
mcdn_error = load_json(mcdn_path)
scrn_error = load_json(scrn_path)
tb_error = load_json(tb_path)

mcdn_ids = set(mcdn_error['FP'] + mcdn_error['FN'])
scrn_ids = set(scrn_error['FP'] + scrn_error['FN'])
tb_ids = set(tb_error['FP'] + tb_error['FN'])

ms_com = mcdn_ids.intersection(scrn_ids)
st_com = scrn_ids.intersection(tb_ids)
mst_dif = st_com - mcdn_ids
tsm_dif = tb_ids - mcdn_ids - scrn_ids
stm_dif = scrn_ids - tb_ids - mcdn_ids
ms_dif = scrn_ids - mcdn_ids
sm_dif = mcdn_ids - scrn_ids
mt_dif = tb_ids - mcdn_ids
print(len(mst_dif), mst_dif)
print(len(tsm_dif), tsm_dif)
print(len(stm_dif), stm_dif)
# print(len(ms_com), ms_com)
# print(len(ms_dif), ms_dif)
# print(len(sm_dif), sm_dif)
# print(len(mt_dif), mt_dif)
print('hello world')
