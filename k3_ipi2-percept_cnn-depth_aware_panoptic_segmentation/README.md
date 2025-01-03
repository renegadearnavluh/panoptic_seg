# Depth-aware Panoptic Segmentation

## Installation
* Clone this repository and create conda environment with *requirements.txt* or *environment.yml* file
* Setup the Cityscapes dataset following [this structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md). Then using [RAFT-Stereo](https://github.com/ndinhtuan/RAFT-Stereo) to create disparity image, then replace the one produced by Semi-Global Matching method.

## Evaluation

    cd my_ps
    python tools/debugs/test_panoptic_fcn_with_cityscapes.py --config-file=path_to_the_config_file.yaml

Our config and the pretrained model are provided via [this link](https://drive.google.com/drive/folders/1oNXpKbGiXmN1lpRnFWs4zbAD6HiTEsKf?usp=sharing).

## Train

    cd my_ps
    python tools/train.py --config-file=path_to_the_config_file.yaml

## Results
Results on Cityscapes dataset. Note: the result of PanopticFCN shown in the below table has higher PQ than the original one, it may be caused by our changed config. The detail config is mentioned in our paper.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">PQ</th>
<th valign="bottom">PQ_th</th>
<th valign="bottom">PQ_st</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Fully_Convolutional_Networks_for_Panoptic_Segmentation_CVPR_2021_paper.pdf">PanopticFCN</td>
<td align="center">R50</td>
<td align="center">60.4</td>
<td align="center">53.6</td>
<td align="center">65.4</td>
</tr>
<tr><td align="left"><a href="https://openaccess.thecvf.com/content/WACV2023/papers/de_Geus_Intra-Batch_Supervision_for_Panoptic_Segmentation_on_High-Resolution_Images_WACV_2023_paper.pdf">PanopticFCN + IBS </td>
<td align="center">R50</td>
<td align="center">60.8</td>
<td align="center">54.7</td>
<td align="center">65.3</td>
</tr>
<tr><td align="left">Ours </td>
<td align="center">R50</td>
<td align="center">62.6</td>
<td align="center">56.2</td>
<td align="center">67.3</td>
</tr>
</tbody></table>

This code is based on the source code of [PanopticFCN-IBS](https://github.com/DdeGeus/PanopticFCN-IBS) project. 