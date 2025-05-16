import os

ckpt_paths = [
    (
        "/data/tested_bed/RoboVLMs/RoboVLMs/checkpoints/kosmos_ph_bridge-post-train.pt",
        "/data/tested_bed/RoboVLMs/RoboVLMs/configs/kosmos_ph_post_train_bridge.json",
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/bridge.bash {} {}".format(ckpt, config))
