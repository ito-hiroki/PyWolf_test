import chainer 
import chainer.links as L
import chainer.functions as F

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=2):
        super(MLP, self).__init__()

        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def estimate_wolf(info, base_info, infer_net):
    # 3. NN使って投票
    #x = tuple(np.array(self.info))
    x = info
    return_result = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)
    y = y.array
    result = y.argmax(axis=1)
<<<<<<< HEAD
    print("estimate_wolf_result")
    print(result)
=======
>>>>>>> 1aaac3ff8612131e0eb5245da0a277a6fb696f1e
    for i in range(len(base_info['statusMap'])):
        if(base_info['statusMap'][str(i+1)] == 'ALIVE' and result[i] == 0):
            idx = i + 1
            return_result.append(idx)
    print(return_result)
    if(return_result != []):
        return return_result
    return -1