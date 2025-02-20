# mscoco
cd demo
python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input_root /media/zeke/project_data/Projects/iPPD/map_recon/out_obs/$SCENE_ID/color \
  --opts MODEL.WEIGHTS ../ckpts/model_final_f07440.pkl

# cd demo
# python demo_ori.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
#   --input /media/zeke/project_data/Projects/iPPD/map_recon/out_obs/$SCENE_ID/color/0.png \
#   --opts MODEL.WEIGHTS ../ckpts/model_final_f07440.pkl

# aed20k

# cd demo
# python demo_ori.py --config-file ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
#   --input /media/zeke/project_data/Projects/iPPD/map_recon/out_obs/$SCENE_ID/color/11.png \
#   --opts MODEL.WEIGHTS ../ckpts/model_final_90ee2d.pkl