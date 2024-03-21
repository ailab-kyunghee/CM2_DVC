import json
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_bbox_distribution(bboxes, save_path):
    # bboxes: [video_num, query_num, 4]
    N, M, C = bboxes.shape
    bboxes = bboxes.transpose(1, 0, 2)

    fig = plt.figure(figsize=(40, 20), dpi=300)
    for m in range(M):
        scores = []
        centers = []
        lengths = []
        n_row, n_col = int(np.ceil(np.sqrt(M))) , int(2 * np.ceil(M / np.sqrt(M)))
        ax = fig.add_subplot(n_row, n_col, 2 * m + 1)
        ax2 = fig.add_subplot(n_row, n_col, 2 * m + 2)
        L = 100
        dstr = np.zeros(L)
        for n in range(N):
            s, e = bboxes[m, n, 0], bboxes[m, n, 1]
            score = bboxes[m, n, 2] * L
            centers.append(0.5 * (s + e) * L)
            lengths.append((e - s) * L)
            scores.append(score)
            s = int(max(0, min(s, 1)) * L)
            e = int(max(0, min(e, 1)) * L)
            dstr[s: e + 1] = dstr[s: e + 1] + 1
        dstr = dstr / N
        ax.plot(np.arange(L), dstr)
        scores, centers, lengths = map(np.array, [scores, centers, lengths])
        flierprops = dict(marker='.', markersize=1, markeredgecolor='r')
        ax2.boxplot([scores, centers, lengths], positions=[1,2,3], flierprops=flierprops, vert=False, patch_artist=False, meanline=False,showmeans=True)
        ax.set_xlim([0, 100])
        ax2.set_xlim([0, 100])
    # pdb.set_trace()
    fig.savefig(save_path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.5)
    plt.close(fig)
    pass


def main(p_input, p_output=None):
    if p_output==None:
        p_output = p_input.rstrip('.json') + '.png'
    d = json.load(open(p_input))['results']
    vid_keys = list(d.keys())
    query_num = max([p_info['query_id'] for p_info in d[vid_keys[0]]]) + 1
    bbox_data = np.zeros((len(d), query_num, 3))
    for i, (vid, info) in enumerate(d.items()):
        sorted_info = sorted(info, key=lambda x: x['query_id'])
        for j, p_info in enumerate(sorted_info):
            s,e = p_info['timestamp']
            duration = p_info['vid_duration']
            conf = p_info['proposal_score']
            bbox_data[i, j, 0] = s / duration
            bbox_data[i, j, 1] = e / duration
            bbox_data[i, j, 2] = conf
    plot_bbox_distribution(bbox_data, p_output)


if __name__ == '__main__':
    # p_input = '/apdcephfs/share_1367250/wybertwang/project/VideoDETR/save/0205_bs1_cl005_t01_bert/prediction/num4917_epoch0.json'
    p_input = sys.argv[1]
    p_output = p_input.rstrip('.json') + '.png'
    main(p_input, p_output)
